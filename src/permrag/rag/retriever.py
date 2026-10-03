"""
    The order here is the whole point of the system:

    1. Ask SpiceDB which documents this user may view.
    2. Hand that list to Qdrant as a filter applied *during* the search.
    3. Assert, on the way out, that every returned chunk is in the permitted
       set.

"""

import logging
import uuid
from collections import Counter
from dataclasses import dataclass

from permrag.config import get_settings
from permrag.exceptions import PermissionSystemError
from permrag.llm import active_embedding_model, embed_query
from permrag.observability.langfuse_client import observe_step
from permrag.permissions.service import PermissionService
from permrag.rag.reranker import rerank_all
from permrag.vectorstore.qdrant import QdrantVectorStore, SearchHit

logger = logging.getLogger(__name__)

# Enough of each rerank candidate to judge its relevance by eye in a trace.
SNIPPET_CHARS = 200

@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    permitted_document_ids: list[str]
    zed_token: str | None
    truncated: bool
    reranked: bool = False
    
class PermissionAwareRetriever:
    def __init__(self, vector_store: QdrantVectorStore, permissions: PermissionService) -> None:
        self._store = vector_store
        self._permissions = permissions
        self._settings = get_settings()
        
    async def retrieve(self, user_id: uuid.UUID, question: str) -> RetrievalResult:
        max_docs = self._settings.max_permitted_documents

        # -- 1. permission lookup ------------------------------------------
        with observe_step("permission-lookup", input={"user_id": str(user_id)}) as span:
            permitted_ids, zed_token = await self._lookup(user_id, max_docs)
            span.update(output={"permitted_document_count": len(permitted_ids)})

        # The permission reads are the last database work until the query is
        # logged, so don't hold a pooled connection through embedding, search,
        # reranking and the LLM call.
        await self._permissions.end_transaction()

        truncated = len(permitted_ids) >= max_docs
        if truncated:
            # The filter would silently omit documents beyond the cap, which
            # looks like a retrieval bug rather than a limit. Surface it.
            logger.warning("permitted document set hit the configured cap",extra={"user_id": str(user_id), "cap": max_docs})

        if not permitted_ids:
            logger.info("no permitted documents", extra={"user_id": str(user_id)})
            return RetrievalResult([], [], zed_token, truncated)
        
        # -- 2. filtered vector search -------------------------------------
        with observe_step(
            "embed-query", as_type="embedding", model=active_embedding_model(), input=question
        ) as span:
            query_vector = await embed_query(question)
            span.update(output={"dimensions": len(query_vector)})

        with observe_step("vector-search", as_type="retriever", input={"question": question}) as span:
            hits = await self._store.search(
                query_vector=query_vector,
                permitted_document_ids=permitted_ids,
                limit=self._settings.retrieval_candidate_limit,
            )
            span.update(
                output={
                    "hit_count": len(hits),
                    "hits": [
                        {
                            "title": hit.title,
                            "page_number": hit.page_number,
                            "embedding_score": round(hit.score, 4),
                        }
                        for hit in hits
                    ],
                }
            )

        # -- 3. defence-in-depth assertion ---------------------------------
        permitted_set = set(permitted_ids)
        verified: list[SearchHit] = []
        for hit in hits:
            if hit.document_id in permitted_set:
                verified.append(hit)
            else:
                logger.error(
                    "PERMISSION LEAK: unpermitted chunk returned by filtered search",
                    extra={
                        "user_id": str(user_id),
                        "document_id": hit.document_id,
                        "chunk_id": hit.chunk_id,
                    },
                )
                
        # -- 4. rerank ------------------------------------------------------
        final_limit = self._settings.retrieval_final_limit

        with observe_step(
            "rerank",
            input={
                "candidate_count": len(verified),
                "top_k": final_limit,
                "min_score": self._settings.reranker_min_score,
            },
        ) as span:
            decisions = await rerank_all(question, verified, final_limit)
            embedding_rank = {hit.chunk_id: position for position, hit in enumerate(verified, start=1)}
            counts = Counter(d.status for d in decisions)
            span.update(
                output={
                    "kept": counts["kept"],
                    "below_floor": counts["below_floor"],
                    "beyond_top_k": counts["beyond_top_k"],
                    # Every candidate, best first -- the rejected ones too, so
                    # the trace shows why each chunk did or did not reach the LLM.
                    "candidates": [
                        {
                            "status": d.status,
                            "rerank_rank": d.rank,
                            "rerank_score": round(d.score, 4) if d.score is not None else None,
                            "embedding_rank": embedding_rank[d.candidate.chunk_id],
                            "embedding_score": round(d.candidate.score, 4),
                            "title": d.candidate.title,
                            "page_number": d.candidate.page_number,
                            "snippet": d.candidate.text[:SNIPPET_CHARS],
                        }
                        for d in decisions
                    ],
                }
            )

        final = [d.candidate for d in decisions if d.status == "kept"]
        reranked = any(d.score is not None for d in decisions)
 
        return RetrievalResult(final, permitted_ids, zed_token, truncated, reranked)
    
    async def _lookup(self, user_id: uuid.UUID, max_docs: int) -> tuple[list[str], str | None]:
        """Fail closed. If authorization cannot be consulted, nothing is returned."""
        try:
            return await self._permissions.list_viewable_document_ids(user_id, max_docs)
        except PermissionSystemError:
            logger.exception(
                "permission lookup failed; failing closed",
                extra={"user_id": str(user_id)},
            )
            raise
        