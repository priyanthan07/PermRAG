"""
    The order here is the whole point of the system:

    1. Ask SpiceDB which documents this user may view.
    2. Hand that list to Qdrant as a filter applied *during* the search.
    3. Assert, on the way out, that every returned chunk is in the permitted
       set.

"""

import logging
import uuid
from dataclasses import dataclass

from permrag.config import get_settings
from permrag.exceptions import PermissionSystemError
from permrag.llm.openai_client import embed_query
from permrag.observability.langfuse_client import get_langfuse
from permrag.permissions.service import PermissionService
from permrag.vectorstore.qdrant import QdrantVectorStore, SearchHit

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    permitted_document_ids: list[str]
    zed_token: str | None
    truncated: bool
    
class PermissionAwareRetriever:
    def __init__(self, vector_store: QdrantVectorStore, permissions: PermissionService) -> None:
        self._store = vector_store
        self._permissions = permissions
        self._settings = get_settings()
        
    async def retrieve(self, user_id: uuid.UUID, question: str) -> RetrievalResult:
        langfuse = get_langfuse()
        max_docs = self._settings.max_permitted_documents
        
        # -- 1. permission lookup ------------------------------------------
        if langfuse is not None:
            with langfuse.start_as_current_observation(name="permission-lookup", as_type="span", input={"user_id": str(user_id)}) as span:
                permitted_ids, zed_token = await self._lookup(user_id, max_docs)
                span.update(output={"permitted_document_count": len(permitted_ids)})
        else:
            permitted_ids, zed_token = await self._lookup(user_id, max_docs)
            
        truncated = len(permitted_ids) >= max_docs
        if truncated:
            # The filter would silently omit documents beyond the cap, which
            # looks like a retrieval bug rather than a limit. Surface it.
            logger.warning("permitted document set hit the configured cap",extra={"user_id": str(user_id), "cap": max_docs})

        if not permitted_ids:
            logger.info("no permitted documents", extra={"user_id": str(user_id)})
            return RetrievalResult([], [], zed_token, truncated)
        
        # -- 2. filtered vector search -------------------------------------
        query_vector = await embed_query(question)

        if langfuse is not None:
            with langfuse.start_as_current_observation(
                name="vector-search", as_type="retriever", input={"question": question}
            ) as span:
                hits = await self._store.search(
                    query_vector=query_vector,
                    permitted_document_ids=permitted_ids,
                    limit=self._settings.retrieval_candidate_limit,
                )
                span.update(output={"hit_count": len(hits)})
        else:
            hits = await self._store.search(
                query_vector=query_vector,
                permitted_document_ids=permitted_ids,
                limit=self._settings.retrieval_candidate_limit,
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

        final = verified[: self._settings.retrieval_final_limit]
        return RetrievalResult(final, permitted_ids, zed_token, truncated)
    
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
        