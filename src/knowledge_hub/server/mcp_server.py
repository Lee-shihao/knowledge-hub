"""MCP server — wires together all components and exposes query_knowledge_base tool.

Uses FlagEmbeddingEmbedder (not OllamaEmbedder) for embeddings.
Supports Bearer token auth via StaticTokenVerifier and IP allowlist middleware.
"""
import structlog
from fastmcp import FastMCP
from qdrant_client import QdrantClient

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.server.tools import create_tools

logger = structlog.get_logger()


def create_mcp_app(settings: Settings) -> FastMCP:
    """Build and configure the FastMCP application.

    Wires together all components: health monitor, embedder, vector store,
    reranker, query engine, and MCP tools.

    Args:
        settings: Application settings.

    Returns:
        Configured FastMCP instance.

    Raises:
        ValueError: If MCP_HOST is not 127.0.0.1 and no auth token is set.
    """
    # Infrastructure clients
    qdrant_client = QdrantClient(settings.QDRANT_URL)

    # Health monitor
    health = HealthMonitor(settings, qdrant_client)

    # Storage
    metadata_mgr = SourceMetadataManager(settings, qdrant_client)
    vector_store = QdrantVectorStore(settings, qdrant_client, metadata_mgr)

    # Retrieval components
    embedder = FlagEmbeddingEmbedder(settings)
    reranker = Reranker(settings)
    query_engine = QueryEngine(settings, embedder, vector_store, reranker)

    # Auth
    if settings.MCP_AUTH_TOKEN:
        from fastmcp.server.auth import StaticTokenVerifier
        verifier = StaticTokenVerifier(
            {settings.MCP_AUTH_TOKEN: {"client_id": "knowledge-hub", "scopes": []}}
        )
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if settings.MCP_HOST != "127.0.0.1":
            raise ValueError(
                "MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set. "
                "Set KH_MCP_AUTH_TOKEN to enable LAN access."
            )
        mcp = FastMCP("knowledge-hub")

    # IP allowlist middleware
    if settings.MCP_ALLOWED_IPS:
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, allowed_ips=None):
                super().__init__(app)
                self.allowed_ips = allowed_ips or []

            async def dispatch(self, request, call_next):
                client_ip = request.client.host if request.client else None
                if client_ip not in self.allowed_ips:
                    return JSONResponse({"error": "Forbidden"}, status_code=403)
                return await call_next(request)

        mcp.add_middleware(Middleware(IPAllowlistMiddleware, allowed_ips=settings.MCP_ALLOWED_IPS))

    # Register tools
    tools = create_tools(settings, query_engine, health)
    for _name, fn in tools.items():
        mcp.add_tool(fn)

    # Store health monitor for startup
    mcp._health = health  # accessed by CLI start command

    return mcp


async def run_mcp_server(settings: Settings):
    """Start the MCP server with health monitoring.

    Args:
        settings: Application settings.
    """
    mcp = create_mcp_app(settings)

    # Start health probe loop
    await mcp._health.start()

    transport = "sse" if settings.MCP_TRANSPORT == "sse" else "streamable-http"
    logger.info(
        "mcp_server_starting",
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        transport=transport,
        auth=bool(settings.MCP_AUTH_TOKEN),
    )

    await mcp.run(
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        transport=transport,
    )
