"""MCP tool wrappers around the QueryEngine.

Provides query_knowledge_base tool with health-gated access:
- Checks model_loaded and qdrant health before querying
- Returns error dict when components are unhealthy
"""
import structlog

from knowledge_hub.config import Settings
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput
from knowledge_hub.server.health import HealthMonitor

logger = structlog.get_logger()


def create_tools(settings: Settings, query_engine: QueryEngine, health: HealthMonitor):
    """Create MCP tool wrappers around the QueryEngine.

    Args:
        settings: Application settings.
        query_engine: QueryEngine instance for executing queries.
        health: HealthMonitor instance for health checks.

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
        # Check health first
        status = await health.get_status()
        if not status.model_loaded:
            return {"error": "Embedding model is not available", "results": [], "query_time_ms": 0}
        if not status.qdrant:
            return {"error": "Knowledge base is not available", "results": [], "query_time_ms": 0}

        q_input = QueryInput(
            query=query,
            top_k=top_k,
            filter_source=filter_source,
            filter_tags=filter_tags,
        )
        result = await query_engine.query(q_input)
        return result.model_dump()

    return {"query_knowledge_base": query_knowledge_base}
