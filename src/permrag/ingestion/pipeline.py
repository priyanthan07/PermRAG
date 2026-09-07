import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from permrag.config import get_settings
from permrag.db.models import Document, DocumentPage
from permrag.exceptions import NotFoundError, ValidationError
from permrag.ingestion.chunker import chunk_page, hash_content
from permrag.llm.openai_client import embed_texts
from permrag.permissions.service import PermissionService
from permrag.vectorstore.qdrant import ChunkPayload, QdrantVectorStore, build_chunk_id

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class PageInput:
    page_number: int
    content: str

@dataclass(slots=True)
class IngestionResult:
    document_id: uuid.UUID
    pages_total: int
    pages_indexed: int
    pages_skipped: int
    pages_removed: int
    chunks_written: int

class IngestionPipeline:
    def __init__(
        self,
        session: AsyncSession,
        vector_store: QdrantVectorStore,
        permissions: PermissionService,
    ) -> None:
        self._session = session
        self._store = vector_store
        self._permissions = permissions
        self._settings = get_settings()

    async def ingest(
        self,
        external_id: str,
        title: str,
        owner_department_id: uuid.UUID,
        pages: list[PageInput],
        actor_id: uuid.UUID | None,
        source_uri: str | None = None,
    ) -> IngestionResult:
        if not pages:
            raise ValidationError("Document must contain at least one page")

        numbers = [p.page_number for p in pages]
        if len(numbers) != len(set(numbers)):
            raise ValidationError("Duplicate page numbers in payload")

        document = await self._get_or_create_document(
            external_id=external_id,
            title=title,
            owner_department_id=owner_department_id,
            source_uri=source_uri,
        )

        document.status = "indexing"
        document.error_message = None
        await self._session.flush()

        # Step 2: permission edge before any content becomes searchable.
        await self._permissions.set_document_owner(
            document_id=document.id,
            department_id=owner_department_id,
            actor_id=actor_id,
        )

        try:
            result = await self._sync_pages(document, pages)
        except Exception as exc:
            document.status = "failed"
            document.error_message = str(exc)[:2000]
            await self._session.flush()
            logger.exception("ingestion failed", extra={"document_id": str(document.id)})
            raise

        document.status = "indexed"
        document.indexed_at = datetime.now(UTC)
        await self._session.flush()

        logger.info(
            "ingestion complete",
            extra={
                "document_id": str(document.id),
                "indexed": result.pages_indexed,
                "skipped": result.pages_skipped,
                "removed": result.pages_removed,
            },
        )
        return result
    
    
    async def _get_or_create_document(
        self,
        external_id: str,
        title: str,
        owner_department_id: uuid.UUID,
        source_uri: str | None,
    ) -> Document:
        existing = await self._session.execute(
            select(Document).where(Document.external_id == external_id)
        )
        document = existing.scalar_one_or_none()

        if document is None:
            document = Document(
                external_id=external_id,
                title=title,
                owner_department_id=owner_department_id,
                source_uri=source_uri,
                status="pending",
            )
            self._session.add(document)
            await self._session.flush()
        else:
            document.title = title
            document.source_uri = source_uri
            document.owner_department_id = owner_department_id

        return document
    
    async def _sync_pages(self, document: Document, pages: list[PageInput]) -> IngestionResult:
        existing_result = await self._session.execute(
            select(DocumentPage).where(DocumentPage.document_id == document.id)
        )
        existing_pages = {p.page_number: p for p in existing_result.scalars().all()}
        incoming_numbers = {p.page_number for p in pages}

        indexed = skipped = removed = chunks_written = 0

        # Pages gone from the source: drop their chunks and their rows.
        for page_number, page_row in existing_pages.items():
            if page_number not in incoming_numbers:
                await self._store.delete_page(str(document.id), page_number)
                await self._session.delete(page_row)
                removed += 1

        for page in pages:
            content_hash = hash_content(page.content)
            existing_page = existing_pages.get(page.page_number)

            if existing_page is not None and existing_page.content_hash == content_hash:
                skipped += 1
                continue

            written = await self._index_page(document, page, content_hash, existing_page)
            chunks_written += written
            indexed += 1

        await self._session.flush()

        return IngestionResult(
            document_id=document.id,
            pages_total=len(pages),
            pages_indexed=indexed,
            pages_skipped=skipped,
            pages_removed=removed,
            chunks_written=chunks_written,
        )
        
    async def _index_page(
        self,
        document: Document,
        page: PageInput,
        content_hash: str,
        existing_page: DocumentPage | None,
    ) -> int:
        chunks = chunk_page(
            page.content,
            chunk_size_words=self._settings.chunk_size_words,
            overlap_words=self._settings.chunk_overlap_words,
        )

        # Clear the old chunks first. Chunk ids are deterministic, so a page
        # that shrank would otherwise leave stale trailing chunks behind.
        await self._store.delete_page(str(document.id), page.page_number)

        if not chunks:
            self._upsert_page_row(document, page, content_hash, 0, existing_page)
            return 0

        version = (existing_page.version + 1) if existing_page else 1
        vectors = await embed_texts([c.text for c in chunks])

        payloads = [
            ChunkPayload(
                chunk_id=build_chunk_id(document.id, page.page_number, c.index),
                document_id=str(document.id),
                department_id=str(document.owner_department_id),
                page_number=page.page_number,
                chunk_index=c.index,
                version=version,
                text=c.text,
                title=document.title,
                source_uri=document.source_uri,
            )
            for c in chunks
        ]

        await self._store.upsert_chunks(payloads, vectors)
        self._upsert_page_row(document, page, content_hash, len(chunks), existing_page, version)
        return len(chunks)
    
    def _upsert_page_row(
        self,
        document: Document,
        page: PageInput,
        content_hash: str,
        chunk_count: int,
        existing_page: DocumentPage | None,
        version: int | None = None,
    ) -> None:
        if existing_page is None:
            self._session.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    content_hash=content_hash,
                    char_count=len(page.content),
                    chunk_count=chunk_count,
                    version=version or 1,
                )
            )
        else:
            existing_page.content_hash = content_hash
            existing_page.char_count = len(page.content)
            existing_page.chunk_count = chunk_count
            if version is not None:
                existing_page.version = version

    async def delete_document(self, document_id: uuid.UUID, actor_id: uuid.UUID | None) -> None:
        """Tear down a document: permissions first, then content."""
        result = await self._session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document is None:
            raise NotFoundError(f"Document {document_id} not found")

        await self._permissions.purge_document_relationships(document_id, actor_id)
        await self._store.delete_document(str(document_id))

        document.status = "deleted"
        await self._session.flush()
        logger.info("document deleted", extra={"document_id": str(document_id)})
        