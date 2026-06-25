"""MCP server — receives AppState and exposes query/list/status tools.

Uses FlagEmbeddingEmbedder (not OllamaEmbedder) for embeddings.
Supports Bearer token auth via StaticTokenVerifier and IP allowlist middleware.
"""
import structlog
from fastmcp import FastMCP

from knowledge_hub.server.app_state import AppState
from knowledge_hub.server.tools import create_tools

logger = structlog.get_logger()


def create_mcp_app(state: AppState) -> FastMCP:
    """Build and configure the FastMCP application from AppState.

    All heavy components (embedder, reranker, qdrant) come from state —
    this function only handles auth, middleware, and tool registration.

    Args:
        state: Fully initialized AppState (mcp may be None).

    Returns:
        Configured FastMCP instance.

    Raises:
        ValueError: If MCP_HOST is not 127.0.0.1 and no auth token is set.
    """
    # Auth setup
    if state.settings.MCP_AUTH_TOKEN:
        from fastmcp.server.auth import StaticTokenVerifier

        verifier = StaticTokenVerifier(
            {
                state.settings.MCP_AUTH_TOKEN: {
                    "client_id": "knowledge-hub",
                    "scopes": [],
                }
            }
        )
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if state.settings.MCP_HOST != "127.0.0.1":
            raise ValueError(
                "MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set. "
                "Set KH_MCP_AUTH_TOKEN to enable LAN access."
            )
        mcp = FastMCP("knowledge-hub")

    # IP allowlist middleware — identical to original implementation
    if state.settings.MCP_ALLOWED_IPS:
        import ipaddress

        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        allowed_networks = []
        for ip_str in state.settings.MCP_ALLOWED_IPS:
            try:
                if "/" in ip_str:
                    allowed_networks.append(
                        ipaddress.ip_network(ip_str, strict=False)
                    )
                else:
                    allowed_networks.append(
                        ipaddress.ip_network(f"{ip_str}/32", strict=False)
                    )
            except ValueError:
                logger.warning("invalid_ip_in_allowlist", ip=ip_str)

        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, allowed_networks=None):
                super().__init__(app)
                self.allowed_networks = allowed_networks or []

            async def dispatch(self, request, call_next):
                client_ip = (
                    request.client.host if request.client else None
                )
                if client_ip:
                    try:
                        ip_addr = ipaddress.ip_address(client_ip)
                        if not any(
                            ip_addr in net
                            for net in self.allowed_networks
                        ):
                            return JSONResponse(
                                {"error": "Forbidden"}, status_code=403
                            )
                    except ValueError:
                        return JSONResponse(
                            {"error": "Forbidden"}, status_code=403
                        )
                else:
                    return JSONResponse(
                        {"error": "Forbidden"}, status_code=403
                    )
                return await call_next(request)

        mcp.add_middleware(
            Middleware(
                IPAllowlistMiddleware, allowed_networks=allowed_networks
            )
        )

    # Register tools — pass components from state
    tools = create_tools(
        state.settings,
        state.query_engine,
        state.health,
        state.metadata_mgr,
        state.vector_store,
    )
    for name, fn in tools.items():
        mcp.add_tool(fn)

    # Store health for test compatibility
    mcp._health = state.health

    return mcp
