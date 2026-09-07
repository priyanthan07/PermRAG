"""Permission boundary tests.

These assertions are deterministic on purpose. Retrieval correctness is a
yes/no security invariant, so it is checked with plain equality rather than an
LLM judge -- a judge with position and verbosity biases is the wrong tool for
a question that must be exactly right every time.

Quality metrics (faithfulness, answer relevancy) belong in a separate suite
and must never be mixed with these.
"""

from unittest.mock import AsyncMock, patch

import pytest

from permrag.exceptions import PermissionSystemError
from permrag.rag.retriever import PermissionAwareRetriever
from permrag.vectorstore.qdrant import SearchHit

FAKE_VECTOR = [0.1] * 1536


def make_hit(chunk_id: str, document_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        department_id="dept-1",
        page_number=1,
        text="content",
        title="Doc",
        score=0.9,
    )


@pytest.fixture
def patched_embed():
    with patch(
        "permrag.rag.retriever.embed_query",
        new=AsyncMock(return_value=FAKE_VECTOR),
    ) as mocked:
        yield mocked


async def test_no_permissions_returns_nothing_and_never_searches(patched_embed):
    """A user with no grants must get zero results, and the vector store must
    not be queried at all -- not queried-then-filtered."""
    store = AsyncMock()
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.return_value = ([], "token-1")

    result = await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    assert result.hits == []
    assert not store.search.called


async def test_permitted_ids_are_passed_as_search_filter(patched_embed):
    """The permitted set must reach Qdrant as a filter, so the constraint is
    applied during the search rather than after it."""
    store = AsyncMock()
    store.search.return_value = [make_hit("c1", "doc-1")]
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.return_value = (["doc-1", "doc-2"], "tok")

    await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    assert store.search.call_args.kwargs["permitted_document_ids"] == ["doc-1", "doc-2"]


async def test_unpermitted_chunk_is_stripped(patched_embed):
    """Defence in depth: if a chunk outside the permitted set ever came back,
    it must not reach the model."""
    store = AsyncMock()
    store.search.return_value = [
        make_hit("c1", "doc-permitted"),
        make_hit("c2", "doc-FORBIDDEN"),
    ]
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.return_value = (["doc-permitted"], "tok")

    result = await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    returned = {hit.document_id for hit in result.hits}
    assert returned == {"doc-permitted"}
    assert "doc-FORBIDDEN" not in returned


async def test_permission_system_failure_fails_closed(patched_embed):
    """If SpiceDB cannot be consulted, retrieval must abort rather than fall
    back to an unfiltered search."""
    store = AsyncMock()
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.side_effect = PermissionSystemError("down")

    with pytest.raises(PermissionSystemError):
        await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    assert not store.search.called


async def test_results_are_capped_at_final_limit(patched_embed):
    store = AsyncMock()
    store.search.return_value = [make_hit(f"c{i}", "doc-1") for i in range(40)]
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.return_value = (["doc-1"], "tok")

    result = await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    from permrag.config import get_settings

    assert len(result.hits) == get_settings().retrieval_final_limit


async def test_truncation_is_reported_when_cap_is_hit(patched_embed):
    """Silently dropping documents past the cap would look like a retrieval
    bug rather than a configured limit."""
    from permrag.config import get_settings

    cap = get_settings().max_permitted_documents
    store = AsyncMock()
    store.search.return_value = []
    permissions = AsyncMock()
    permissions.list_viewable_document_ids.return_value = (
        [f"doc-{i}" for i in range(cap)],
        "tok",
    )

    result = await PermissionAwareRetriever(store, permissions).retrieve("user-1", "q")

    assert result.truncated is True
    