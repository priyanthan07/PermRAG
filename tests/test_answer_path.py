"""Answer generation, citations, empty model output and the embedding check."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from permrag.exceptions import LLMError, VectorStoreError
from permrag.rag.answerer import Answerer, cited_indices
from permrag.rag.retriever import RetrievalResult
from permrag.vectorstore.qdrant import QdrantVectorStore, SearchHit


def hit(n: int) -> SearchHit:
    return SearchHit(
        chunk_id=f"c{n}", document_id=f"d{n}", department_id="x", page_number=n,
        text=f"passage {n}", title=f"Doc {n}", score=0.5,
    )


def test_cited_indices_reads_single_and_grouped_markers():
    assert cited_indices("A [1]. B [2, 4]. C [3][5].") == {1, 2, 3, 4, 5}
    assert cited_indices("no citations here") == set()


async def _answer(model_text: str):
    retriever = AsyncMock()
    retriever.retrieve.return_value = RetrievalResult([hit(1), hit(2), hit(3)], ["d1", "d2", "d3"], None, False)
    session = MagicMock()
    session.flush = AsyncMock()
    with (
        patch("permrag.rag.answerer.generate_answer", AsyncMock(return_value=(model_text, {}))),
        patch("permrag.rag.answerer.asyncio.create_task", MagicMock()),
    ):
        return await Answerer(retriever, session).answer(uuid.uuid4(), "q")


async def test_only_cited_passages_are_returned_as_citations():
    result = await _answer("Revenue beat plan [2].")
    assert [c.index for c in result.citations] == [2]
    assert result.retrieved_chunk_count == 3  # all three still reached the model


async def test_all_passages_are_returned_when_none_is_cited():
    result = await _answer("Revenue beat plan.")
    assert [c.index for c in result.citations] == [1, 2, 3]


# --- empty model output --------------------------------------------------------


async def test_gemini_empty_text_is_an_error_not_an_empty_answer():
    from permrag.llm import gemini_client

    response = SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")], usage_metadata=None)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response))))
    with patch.object(gemini_client, "get_gemini_client", return_value=client), pytest.raises(LLMError, match="MAX_TOKENS"):
        await gemini_client.generate_answer("system", "user")


async def test_openai_empty_text_is_an_error_not_an_empty_answer():
    from permrag.llm import openai_client

    choice = SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="content_filter")
    response = SimpleNamespace(choices=[choice], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    with patch.object(openai_client, "get_openai_client", return_value=client), pytest.raises(LLMError, match="content_filter"):
        await openai_client.generate_answer("system", "user")


# --- embedding width vs stored collection -----------------------------------------


async def test_collection_with_a_different_vector_width_fails_startup():
    store = QdrantVectorStore()
    store._dimensions = 1536
    store._client = AsyncMock()
    store._client.collection_exists.return_value = True
    store._client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(size=3072, distance=models.Distance.COSINE)))
    )
    with pytest.raises(VectorStoreError, match="3072-dim vectors, but the configured embedding model produces 1536"):
        await store.ensure_collection()
    assert not store._client.create_collection.called
