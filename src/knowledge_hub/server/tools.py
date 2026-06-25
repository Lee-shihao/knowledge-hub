"""MCP tool wrappers around the QueryEngine.

Provides 3 tools:
- query_knowledge_base: semantic search with health-gated access
- list_kb_sources: list all indexed source files with metadata
- get_kb_status: system health and collection statistics
"""
import structlog

from knowledge_hub.config import Settings
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.storage.vector_store import QdrantVectorStore

logger = structlog.get_logger()


def create_tools(
    settings: Settings,
    query_engine: QueryEngine,
    health: HealthMonitor,
    metadata_mgr: SourceMetadataManager,
    vector_store: QdrantVectorStore,
) -> dict:
    """Create MCP tool wrappers.

    Args:
        settings: Application settings.
        query_engine: QueryEngine instance for executing queries.
        health: HealthMonitor instance for health checks.
        metadata_mgr: SourceMetadataManager for listing sources.
        vector_store: QdrantVectorStore for collection stats.

    Returns:
        Dict mapping tool name to async callable.
    """

    async def query_knowledge_base(
        query: str,
        top_k: int = 5,
        filter_source: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> dict:
        """Search the knowledge base for relevant document chunks.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default 5).
            filter_source: Optional filter by source filename.
            filter_tags: Optional filter by document tags.

        Returns:
            Dictionary with 'results' list and 'query_time_ms' float.
        """
        status = await health.get_status()
        if not status.model_loaded:
            return {
                "error": "Embedding model is not available",
                "results": [],
                "query_time_ms": 0,
            }
        if not status.qdrant:
            return {
                "error": "Knowledge base is not available",
                "results": [],
                "query_time_ms": 0,
            }

        q_input = QueryInput(
            query=query,
            top_k=top_k,
            filter_source=filter_source,
            filter_tags=filter_tags,
        )
        result = await query_engine.query(q_input)
        return result.model_dump()

    async def list_kb_sources() -> dict:
        """List all indexed source files with metadata.

        Returns:
            Dictionary with 'sources' list and 'count' int.
        """
        details = await metadata_mgr.list_source_details()
        sources = [
            {
                "filename": d["source_file"],
                "chunk_count": d["chunk_count"],
                "source_hash": d["source_hash"],
            }
            for d in details
        ]
        return {"sources": sources, "count": len(sources)}

    async def get_kb_status() -> dict:
        """Get knowledge base system status and statistics.

        Returns:
            Dictionary with health status and collection statistics.
        """
        h = await health.get_status()
        try:
            total_chunks = await vector_store.count()
            sources = await metadata_mgr.list_sources()
            total_sources = len(sources)
        except Exception:
            total_chunks = -1
            total_sources = -1

        return {
            "model_loaded": h.model_loaded,
            "qdrant": h.qdrant,
            "gpu_available": h.gpu_available,
            "gpu_memory_free_mb": h.gpu_memory_free_mb,
            "collection": settings.QDRANT_COLLECTION,
            "total_chunks": total_chunks,
            "total_sources": total_sources,
        }

    return {
        "query_knowledge_base": query_knowledge_base,
        "list_kb_sources": list_kb_sources,
        "get_kb_status": get_kb_status,
    }
