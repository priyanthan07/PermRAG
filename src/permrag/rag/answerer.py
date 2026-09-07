import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from permrag.db.models import QueryLog
from permrag.llm.openai_client import generate_answer
from permrag.observability.langfuse_client import get_langfuse, get_trace_id
from permrag.rag.retriever import PermissionAwareRetriever
from permrag.vectorstore.qdrant import SearchHit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
    You are an internal knowledge assistant for a large organization.

    Rules:
    - Answer only from the numbered passages provided. They are the complete set of sources available for this question.
    - If the passages do not contain the answer, say so plainly. Never fill the gap with general knowledge or assumption.
    - Cite the passages you used as [1], [2], and so on.
    - Be concise and factual. Do not speculate about material you cannot see.
"""

NO_CONTEXT_MESSAGE = (
    "I could not find anything you have access to that answers this question. "
    "If you believe you should have access to relevant material, contact your "
    "administrator."
)

@dataclass(slots=True)
class Citation:
    index: int
    document_id: str
    title: str
    page_number: int
    source_uri: str | None


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    permitted_document_count: int
    retrieved_chunk_count: int
    trace_id: str | None
    latency_ms: int
    
def build_context(hits: list[SearchHit]) -> tuple[str, list[Citation]]:
    blocks: list[str] = []
    citations: list[Citation] = []

    for position, hit in enumerate(hits, start=1):
        blocks.append(f"[{position}] Title: {hit.title} (page {hit.page_number})\n{hit.text}")
        citations.append(
            Citation(
                index=position,
                document_id=hit.document_id,
                title=hit.title,
                page_number=hit.page_number,
                source_uri=hit.source_uri,
            )
        )

    return "\n\n".join(blocks), citations

class Answerer:
    def __init__(
        self,
        retriever: PermissionAwareRetriever,
        session: AsyncSession,
    ) -> None:
        self._retriever = retriever
        self._session = session
        
    async def answer(self, user_id: uuid.UUID, question: str) -> AnswerResult:
        started = datetime.now(UTC)
        langfuse = get_langfuse()

        retrieval = await self._retriever.retrieve(user_id, question)

        if not retrieval.hits:
            result = AnswerResult(
                answer=NO_CONTEXT_MESSAGE,
                citations=[],
                permitted_document_count=len(retrieval.permitted_document_ids),
                retrieved_chunk_count=0,
                trace_id=get_trace_id(),
                latency_ms=self._elapsed_ms(started),
            )
            await self._log_query(user_id, question, retrieval, result)
            return result

        context, citations = build_context(retrieval.hits)
        user_prompt = f"Passages:\n\n{context}\n\nQuestion: {question}"

        if langfuse is not None:
            with langfuse.start_as_current_observation(name="generate-answer", as_type="generation", input={"question": question}) as span:
                answer_text, usage = await generate_answer(SYSTEM_PROMPT, user_prompt)
                span.update(output=answer_text, usage_details=usage)
        else:
            answer_text, _ = await generate_answer(SYSTEM_PROMPT, user_prompt)

        result = AnswerResult(
            answer=answer_text,
            citations=citations,
            permitted_document_count=len(retrieval.permitted_document_ids),
            retrieved_chunk_count=len(retrieval.hits),
            trace_id=get_trace_id(),
            latency_ms=self._elapsed_ms(started),
        )
        await self._log_query(user_id, question, retrieval, result)
        return result

    @staticmethod
    def _elapsed_ms(started: datetime) -> int:
        return int((datetime.now(UTC) - started).total_seconds() * 1000)
    
    async def _log_query(
        self,
        user_id: uuid.UUID,
        question: str,
        retrieval,  # RetrievalResult
        result: AnswerResult,
    ) -> None:
        """Record what the user was permitted to see and what actually reached
        the model, so a permission incident can be reconstructed later."""
        self._session.add(
            QueryLog(
                created_at=datetime.now(UTC),
                user_id=user_id,
                question=question,
                answer=result.answer,
                permitted_document_count=result.permitted_document_count,
                # Cap the stored list: a user permitted to see everything would
                # otherwise write a very large array on every single query.
                permitted_document_ids=retrieval.permitted_document_ids[:200],
                retrieved_chunk_ids=[h.chunk_id for h in retrieval.hits],
                zed_token=retrieval.zed_token,
                trace_id=result.trace_id,
                latency_ms=result.latency_ms,
            )
        )
        await self._session.flush()
        