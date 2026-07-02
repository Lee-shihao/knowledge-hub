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
        """List all distinct source filenames from the main vector collection.

        Queries the main collection directly instead of the metadata collection
        to ensure an always-accurate view of what is actually stored.
        """
        main_collection = self.settings.QDRANT_COLLECTION
        collections = [c.name for c in self._client.get_collections().collections]
        if main_collection not in collections:
            return []

        sources: set[str] = set()
        points, next_offset = self._client.scroll(
            collection_name=main_collection,
            limit=1000,
            with_payload=["source_file"],
            with_vectors=False,
        )
        for p in points:
            if p.payload and "source_file" in p.payload:
                sources.add(p.payload["source_file"])

        while next_offset:
            points, next_offset = self._client.scroll(
                collection_name=main_collection,
                offset=next_offset,
                limit=1000,
                with_payload=["source_file"],
                with_vectors=False,
            )
            for p in points:
                if p.payload and "source_file" in p.payload:
                    sources.add(p.payload["source_file"])

        return sorted(sources)

    async def list_source_details(self) -> list[dict]:
        """Return aggregated source details from the main vector collection.

        Scrolls the main collection and aggregates by source_file to produce
        an always-accurate list, independent of the _source_meta metadata
        collection.  The metadata collection is only used for fast hash
        lookups during incremental ingestion.

        Returns:
            List of dicts with source_file, source_hash, and chunk_count.
        """
        main_collection = self.settings.QDRANT_COLLECTION
        collections = [c.name for c in self._client.get_collections().collections]
        if main_collection not in collections:
            return []

        source_map: dict[str, dict] = {}
        points, next_offset = self._client.scroll(
            collection_name=main_collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            sf = (p.payload or {}).get("source_file", "unknown")
            if sf not in source_map:
                source_map[sf] = {
                    "source_file": sf,
                    "source_hash": (p.payload or {}).get("source_hash", ""),
                    "chunk_count": 0,
                }
            source_map[sf]["chunk_count"] += 1

        while next_offset:
            points, next_offset = self._client.scroll(
                collection_name=main_collection,
                offset=next_offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                sf = (p.payload or {}).get("source_file", "unknown")
                if sf not in source_map:
                    source_map[sf] = {
                        "source_file": sf,
                        "source_hash": (p.payload or {}).get("source_hash", ""),
                        "chunk_count": 0,
                    }
                source_map[sf]["chunk_count"] += 1

        return list(source_map.values())

    async def orphan_cleanup(self, local_source_files: set[str]) -> int:
        """Remove vectors and metadata for files no longer on disk.

        Deletes from both the main vector collection and the _source_meta
        metadata collection.  Returns the number of orphaned sources removed.
        """
        main_collection = self.settings.QDRANT_COLLECTION
        collections = [c.name for c in self._client.get_collections().collections]

        db_sources = set(await self.list_sources())
        orphans = db_sources - local_source_files
        for orphan in orphans:
            # Delete from main vector collection
            if main_collection in collections:
                self._client.delete(
                    collection_name=main_collection,
                    points_selector=Filter(
                        must=[FieldCondition(key="source_file", match=MatchValue(value=orphan))]
                    ),
                )
            # Delete from metadata collection
            if self._collection in collections:
                await self.remove(orphan)
        return len(orphans)
