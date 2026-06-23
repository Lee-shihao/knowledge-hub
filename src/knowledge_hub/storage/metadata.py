from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from knowledge_hub.config import Settings


class SourceMetadataManager:
    """Manages source file metadata in a separate Qdrant collection.

    Uses a lightweight collection keyed by source_file to track
    content hashes for incremental updates and enable O(1) lookups
    without scanning all vector points.
    """

    def __init__(self, settings: Settings, client: QdrantClient):
        self.settings = settings
        self._collection = f"{settings.QDRANT_COLLECTION}_source_meta"
        self._client = client

    async def ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    async def get_hash(self, source_file: str) -> str | None:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
            limit=1,
        )
        if points:
            return points[0].payload.get("source_hash")
        return None

    async def upsert(self, source_file: str, source_hash: str, chunk_count: int) -> None:
        import hashlib

        point_id = int(hashlib.md5(source_file.encode()).hexdigest()[:16], 16) % (2**63)
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload={
                        "source_file": source_file,
                        "source_hash": source_hash,
                        "chunk_count": chunk_count,
                    },
                )
            ],
        )

    async def remove(self, source_file: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )

    async def list_sources(self) -> list[str]:
        points, next_offset = self._client.scroll(
            collection_name=self._collection, limit=100
        )
        sources = [p.payload["source_file"] for p in points]
        while next_offset:
            points, next_offset = self._client.scroll(
                collection_name=self._collection, offset=next_offset, limit=100
            )
            sources.extend(p.payload["source_file"] for p in points)
        return sources

    async def orphan_cleanup(self, local_source_files: set[str]) -> int:
        db_sources = set(await self.list_sources())
        orphans = db_sources - local_source_files
        for orphan in orphans:
            await self.remove(orphan)
        return len(orphans)