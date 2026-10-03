"""Chat route: the permission-filtered question answering endpoint."""

import logging
from contextlib import AbstractContextManager, nullcontext
from typing import Annotated

from fastapi import APIRouter, Depends

from permrag.api.deps import CurrentUser, get_answerer
from permrag.api.schemas import ChatRequest, ChatResponse, CitationResponse
from permrag.config import QUESTION_TOKEN_RESERVE, get_settings
from permrag.exceptions import ValidationError
from permrag.ingestion.chunker import count_tokens
from permrag.observability.langfuse_client import (
    get_langfuse,
    get_trace_id,
    observe_step,
)
from permrag.rag.answerer import Answerer, record_failed_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

AnswererDep = Annotated[Answerer, Depends(get_answerer)]


@router.post("", response_model=ChatResponse)
async def ask(
    payload: ChatRequest,
    user: CurrentUser,
    answerer: AnswererDep,
) -> ChatResponse:
    """Answer a question using only documents this user is permitted to see.

    The user id comes from the verified bearer token, never from the request
    body -- otherwise a caller could ask questions as somebody else.
    """
    if get_settings().reranker_enabled:
        # Chunks are sized so that chunk + question fits the reranker window.
        # A longer question would make the reranker cut the passage short.
        question_tokens = count_tokens(payload.question)
        if question_tokens > QUESTION_TOKEN_RESERVE:
            raise ValidationError(
                f"Question is too long ({question_tokens} tokens; at most {QUESTION_TOKEN_RESERVE}). "
                "Please shorten it."
            )

    trace_attributes: AbstractContextManager[object] = nullcontext()
    if get_langfuse() is not None:
        from langfuse import propagate_attributes

        trace_attributes = propagate_attributes(user_id=str(user.id), trace_name="permrag-chat")

    with trace_attributes:
        with observe_step("permrag-chat", input={"question": payload.question}) as span:
            try:
                result = await answerer.answer(user_id=user.id, question=payload.question)
            except Exception as exc:
                await record_failed_query(user.id, payload.question, exc, get_trace_id())
                raise
            span.update(
                output={
                    "answer": result.answer,
                    "permitted_document_count": result.permitted_document_count,
                    "retrieved_chunk_count": result.retrieved_chunk_count,
                }
            )

    return ChatResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                index=c.index,
                document_id=c.document_id,
                title=c.title,
                page_number=c.page_number,
                source_uri=c.source_uri,
            )
            for c in result.citations
        ],
        permitted_document_count=result.permitted_document_count,
        permitted_documents_truncated=result.permitted_documents_truncated,
        retrieved_chunk_count=result.retrieved_chunk_count,
        trace_id=result.trace_id,
        latency_ms=result.latency_ms,
    )
