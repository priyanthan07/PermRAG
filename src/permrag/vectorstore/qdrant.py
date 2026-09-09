import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from permrag.config import get_settings
from permrag.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

# Payload keys. Anything the permission filter touches must be indexed.
FIELD_DOCUMENT_ID = "document_id"
FIELD_DEPARTMENT_ID = "department_id"
FIELD_PAGE_NUMBER = "page_number"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_VERSION = "version"
FIELD_TEXT = "text"
FIELD_TITLE = "title"
FIELD_SOURCE_URI = "source_uri"

@dataclass(slots=True)
class ChunkPayload:
    """One embedded passage plus the metadata retrieval filters on."""

    chunk_id: str
    document_id: str
    department_id: str
    page_number: int
    chunk_index: int
    version: int
    text: str
    title: str
    source_uri: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            FIELD_DOCUMENT_ID: self.document_id,
            FIELD_DEPARTMENT_ID: self.department_id,
            FIELD_PAGE_NUMBER: self.page_number,
            FIELD_CHUNK_INDEX: self.chunk_index,
            FIELD_VERSION: self.version,
            FIELD_TEXT: self.text,
            FIELD_TITLE: self.title,
            FIELD_SOURCE_URI: self.source_uri,
        }


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    department_id: str
    page_number: int
    text: str
    title: str
    score: float
    source_uri: str | None = None
    
class QdrantVectorStore:
    """Async Qdrant wrapper. One instance per process."""
    
    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        self._client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key or None, timeout=30)
        self._collection = settings.qdrant_collection
        self._dimensions = settings.openai_embedding_dimensions
        logger.info("qdrant client initialised",extra={"url": settings.qdrant_url, "collection": self._collection})
    
    @property
    def collection_name(self) -> str:
        return self._collection
    
    # -- lifecycle ---------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the collection and payload indexes if absent. Idempotent."""
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("qdrant collection created", extra={"collection": self._collection})

            # Keyword indexes on the filtered fields. Without these, Qdrant
            # scans payloads instead of using the index, and filtered search
            # degrades badly as the collection grows.
            for field in (FIELD_DOCUMENT_ID, FIELD_DEPARTMENT_ID):
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to prepare collection: {exc}") from exc

    async def close(self) -> None:
        await self._client.close()
        
    # -- writes ------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[ChunkPayload], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError(f"Chunk upsert failed: {exc}") from exc

        logger.info("chunks upserted", extra={"count": len(points)})
        
    async def delete_document(self, document_id: str) -> None:
        """Remove every chunk of a document."""
        await self._delete_by_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key=FIELD_DOCUMENT_ID,
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        )
        logger.info("document chunks deleted", extra={"document_id": document_id})
        
    async def delete_page(self, document_id: str, page_number: int) -> None:
        """Remove chunks for one page, ahead of re-embedding it."""
        await self._delete_by_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key=FIELD_DOCUMENT_ID,
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key=FIELD_PAGE_NUMBER,
                        match=models.MatchValue(value=page_number),
                    ),
                ]
            )
        )
        
    async def _delete_by_filter(self, flt: models.Filter) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Chunk deletion failed: {exc}") from exc
        
    # -- reads -------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        permitted_document_ids: list[str],
        limit: int,
    ) -> list[SearchHit]:
        """
            Vector search constrained to permitted documents.

            ``permitted_document_ids`` is not optional and is not a hint. It
            becomes a ``must`` condition inside the query, so Qdrant never scores
            a chunk the user cannot see.
        """
        if not permitted_document_ids:
            # No permissions means no results. Never fall through to an unfiltered search.
            logger.info("search skipped: empty permitted set")
            return []
        
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=FIELD_DOCUMENT_ID,
                    match=models.MatchAny(any=permitted_document_ids),
                )
            ]
        )
        
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc
        
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    document_id=str(payload.get(FIELD_DOCUMENT_ID, "")),
                    department_id=str(payload.get(FIELD_DEPARTMENT_ID, "")),
                    page_number=int(payload.get(FIELD_PAGE_NUMBER, 0)),
                    text=str(payload.get(FIELD_TEXT, "")),
                    title=str(payload.get(FIELD_TITLE, "")),
                    source_uri=payload.get(FIELD_SOURCE_URI),
                    score=float(point.score),
                )
            )

        logger.info(
            "vector search complete",
            extra={"hits": len(hits), "permitted_documents": len(permitted_document_ids)},
        )
        return hits
    
    async def count_document_chunks(self, document_id: str) -> int:
        try:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=FIELD_DOCUMENT_ID,
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
            return result.count
        except Exception as exc:
            raise VectorStoreError(f"Chunk count failed: {exc}") from exc
        

def build_chunk_id(document_id: uuid.UUID | str, page_number: int, chunk_index: int) -> str:
    """
        Deterministic chunk id.

        Re-embedding the same page produces the same ids, so an upsert overwrites
        in place instead of accumulating orphans.
    """
    seed = f"{document_id}:{page_number}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore()
    return _store
