from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Prefetch,
    FusionQuery,
)

from knowledge_hub.config import Settings
from knowledge_hub.schemas import DocumentChunk
from knowledge_hub.storage.metadata import SourceMetadataManager


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a QdrantClient based on QDRANT_MODE setting.

    Embedded mode: creates the storage directory and returns a path-based client.
    HTTP mode: returns a url-based client with compatibility checks disabled.

    Args:
        settings: Application settings.

    Returns:
        Configured QdrantClient instance.
    """
    if settings.QDRANT_MODE == "embedded":
        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=settings.QDRANT_PATH)
    return QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)


class QdrantVectorStore:
    """Manages the Qdrant vector collection for document chunks.

    Stores dense (1024d) and sparse vectors natively in Qdrant,
    enabling server-side hybrid search via RRF.
    """

    def __init__(self, settings: Settings, client: QdrantClient, metadata_mgr: SourceMetadataManager):
        self.settings = settings
        self._collection = settings.QDRANT_COLLECTION
        self._client = client
        self._metadata = metadata_mgr

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, with dense+sparse vectors."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection in collections:
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Upsert document chunks with both dense and sparse embeddings."""
        points = []
        for chunk in chunks:
            sparse_indices = list(chunk.sparse_embedding.keys())
            sparse_values = [chunk.sparse_embedding[i] for i in sparse_indices]
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": chunk.dense_embedding,
                        "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                    },
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.metadata.source_file,
                        "source_hash": chunk.metadata.source_hash,
                        "page_number": chunk.metadata.page_number,
                        "heading_path": chunk.metadata.heading_path,
                        "tags": chunk.metadata.tags,
                    },
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    async def hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        top_k: int = 20,
        filter_source: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Perform hybrid search using dense and sparse vectors with RRF fusion.

        Args:
            dense_vec: Dense embedding vector (1024d)
            sparse_vec: Sparse embedding as {token_id: score}
            top_k: Number of results to return
            filter_source: Optional filter by source file
            filter_tags: Optional filter by tags (any match)

        Returns:
            List of (chunk_id, score, payload) tuples
        """
        # Build optional payload filter
        must_conditions = []
        if filter_source:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=filter_source))
            )
        if filter_tags:
            must_conditions.append(FieldCondition(key="tags", match=MatchAny(any=filter_tags)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        sparse_indices = list(sparse_vec.keys())
        sparse_values = [sparse_vec[i] for i in sparse_indices]

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=top_k),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            query_filter=query_filter,
            limit=top_k,
        )
        return [(p.id, p.score, p.payload) for p in results.points]

    async def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks from a specific source file.

        Silently succeeds if the collection doesn't exist.
        """
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )

    async def count(self) -> int:
        """Return the total number of vectors in the collection."""
        info = self._client.count(collection_name=self._collection)
        return info.count
