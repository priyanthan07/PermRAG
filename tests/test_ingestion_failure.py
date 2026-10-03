"""A failed ingest must leave a known state and never touch the permission graph."""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from permrag.db.models import Document
from permrag.exceptions import LLMError
from permrag.ingestion.pipeline import IngestionPipeline, PageInput


def fake_session(existing: Document | None):
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    result.scalars.return_value.all.return_value = []
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    # Stand in for the database assigning the primary key on flush.
    session.add.side_effect = lambda obj: setattr(obj, "id", uuid.uuid4())
    return session


async def ingest(session, store, permissions, owner):
    with patch("permrag.ingestion.pipeline.embed_texts", AsyncMock(side_effect=LLMError("embedding down"))):
        await IngestionPipeline(session, store, permissions).ingest(
            external_id="ext-1", title="Doc", owner_department_id=owner,
            pages=[PageInput(1, "some page text")], actor_id=None,
        )


async def test_failed_new_document_leaves_no_edge_and_no_chunks():
    session, store, permissions = fake_session(None), AsyncMock(), AsyncMock()

    with pytest.raises(LLMError):
        await ingest(session, store, permissions, uuid.uuid4())

    assert not permissions.set_document_owner.called, "no ownership edge for a failed ingest"
    session.rollback.assert_awaited()
    store.delete_document.assert_awaited_once()


async def test_failed_existing_document_is_marked_failed_and_keeps_its_owner():
    owner = uuid.uuid4()
    existing = Document(id=uuid.uuid4(), external_id="ext-1", title="Doc", owner_department_id=owner, status="indexed")
    session, store, permissions = fake_session(existing), AsyncMock(), AsyncMock()
    row = SimpleNamespace(status="indexed", error_message=None)

    @asynccontextmanager
    async def failure_scope():
        yield SimpleNamespace(get=AsyncMock(return_value=row))

    with patch("permrag.ingestion.pipeline.session_scope", failure_scope), pytest.raises(LLMError):
        await ingest(session, store, permissions, owner)

    assert not permissions.set_document_owner.called
    session.rollback.assert_awaited()  # before the separate write, to release the row lock
    assert row.status == "failed" and "embedding down" in row.error_message
    assert not store.delete_document.called, "an existing document keeps its chunks"
    assert not store.delete_page.called, "embedding failed, so no page's chunks may be removed"
