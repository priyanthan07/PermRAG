"""Vector storage. Retrieval is always permission-filtered at query time."""

from permrag.vectorstore.qdrant import (
    ChunkPayload,
    QdrantVectorStore,
    SearchHit,
    build_chunk_id,
    get_vector_store,
)

__all__ = ["ChunkPayload", "QdrantVectorStore", "SearchHit", "build_chunk_id", "get_vector_store"]