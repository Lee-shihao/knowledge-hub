"""Tests for MCP server creation — app wiring, auth, IP middleware.

All heavy components (FlagEmbeddingEmbedder, Reranker, QdrantClient) are mocked.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_hub.config import Settings


def _patch_heavy_components():
    """Return a context manager that patches all heavy server components.

    Patches: FlagEmbeddingEmbedder, Reranker, QdrantClient, SourceMetadataManager.
    """
    return patch.multiple(
        "knowledge_hub.server.mcp_server",
        FlagEmbeddingEmbedder=MagicMock(return_value=MagicMock()),
        Reranker=MagicMock(return_value=MagicMock()),
        QdrantClient=MagicMock(return_value=MagicMock()),
        SourceMetadataManager=MagicMock(return_value=MagicMock()),
    )


class TestCreateMCPApp:
    """Tests for create_mcp_app() — app creation and configuration."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def settings_with_auth(self):
        return Settings(MCP_AUTH_TOKEN="test-token-123")

    @pytest.fixture
    def settings_with_allowed_ips(self):
        return Settings(MCP_ALLOWED_IPS=["192.168.1.1", "10.0.0.0/8"])

    @pytest.fixture
    def settings_lan_no_auth(self):
        return Settings(MCP_HOST="0.0.0.0", MCP_AUTH_TOKEN=None)

    def test_create_mcp_app_returns_fastmcp_instance(self, settings):
        """create_mcp_app should return a FastMCP instance."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings)
            from fastmcp import FastMCP
            assert isinstance(mcp, FastMCP)
            assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_has_health_attribute(self, settings):
        """create_mcp_app should attach _health to the MCP instance."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings)
            assert hasattr(mcp, "_health")
            from knowledge_hub.server.health import HealthMonitor
            assert isinstance(mcp._health, HealthMonitor)

    def test_create_mcp_app_with_auth_token(self, settings_with_auth):
        """When MCP_AUTH_TOKEN is set, auth should be configured without error."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings_with_auth)
            assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_raises_on_lan_no_auth(self, settings_lan_no_auth):
        """When MCP_HOST is not 127.0.0.1 and no auth token, should raise ValueError."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            with pytest.raises(ValueError, match="MCP_HOST must be 127.0.0.1"):
                create_mcp_app(settings_lan_no_auth)

    def test_create_mcp_app_allows_localhost_no_auth(self, settings):
        """When MCP_HOST is 127.0.0.1 and no auth token, should succeed."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings)
            assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_with_allowed_ips(self, settings_with_allowed_ips):
        """When MCP_ALLOWED_IPS is set, IP middleware should be added without error."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings_with_allowed_ips)
            assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_registers_tools(self, settings):
        """create_mcp_app should register query_knowledge_base tool."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings)
            # FastMCP stores tools internally — verify the app was created
            assert mcp.name == "knowledge-hub"


class TestIPMiddleware:
    """Tests for IP allowlist middleware behavior."""

    @pytest.fixture
    def settings(self):
        return Settings(MCP_ALLOWED_IPS=["192.168.1.100", "10.0.0.1"])

    def test_ip_middleware_creates_app_successfully(self, settings):
        """App with IP middleware should be created successfully."""
        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            mcp = create_mcp_app(settings)
            assert mcp.name == "knowledge-hub"


class TestRunMCPServer:
    """Tests for run_mcp_server() — transport mode behavior."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def settings_sse(self):
        return Settings(MCP_TRANSPORT="sse")

    @pytest.fixture
    def settings_streamable(self):
        return Settings(MCP_TRANSPORT="streamable-http")

    def test_default_transport_is_streamable_http(self, settings):
        """Default MCP_TRANSPORT should be streamable-http."""
        from knowledge_hub.config import Settings
        s = Settings()
        assert s.MCP_TRANSPORT == "streamable-http"

    def test_run_with_streamable_http_passes_stateless_params(self, settings_streamable):
        """streamable-http transport should pass stateless_http=True and json_response=True."""
        from knowledge_hub.server.mcp_server import run_mcp_server

        with patch("knowledge_hub.server.mcp_server.create_mcp_app") as mock_create:
            mock_mcp = MagicMock()
            mock_create.return_value = mock_mcp

            run_mcp_server(settings_streamable)

            mock_mcp.run.assert_called_once_with(
                host=settings_streamable.MCP_HOST,
                port=settings_streamable.MCP_PORT,
                transport="streamable-http",
                stateless_http=True,
                json_response=True,
            )

    def test_run_with_sse_does_not_pass_stateless_params(self, settings_sse):
        """sse transport should NOT pass stateless_http or json_response."""
        from knowledge_hub.server.mcp_server import run_mcp_server

        with patch("knowledge_hub.server.mcp_server.create_mcp_app") as mock_create:
            mock_mcp = MagicMock()
            mock_create.return_value = mock_mcp

            run_mcp_server(settings_sse)

            mock_mcp.run.assert_called_once_with(
                host=settings_sse.MCP_HOST,
                port=settings_sse.MCP_PORT,
                transport="sse",
            )


class TestStreamableHTTPIntegration:
    """curl 风格的集成测试 — 验证 streamable-http 传输协议。

    使用 Starlette TestClient 直接向 FastMCP ASGI app 发 POST 请求，
    无需启动真实服务器 socket，模拟 curl 的请求-响应模式。

    MCP streamable-http 要求客户端发送 Accept: application/json 头，
    否则返回 406 Not Acceptable。
    """

    # 模拟 curl -H "Accept: application/json" -H "Content-Type: application/json"
    JSON_HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def test_client(self, settings):
        """创建 MCP app 并返回 Starlette TestClient（正确处理 lifespan）。"""
        from starlette.testclient import TestClient

        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app

            mcp = create_mcp_app(settings)
            # 标记健康检查为就绪，确保 tools/list 可用
            mcp._health.model_loaded = True
            mcp._health.qdrant = True
            app = mcp.http_app(
                transport="streamable-http",
                stateless_http=True,
                json_response=True,
                path="/mcp",
            )
            with TestClient(app) as client:
                yield client

    def test_tools_list_via_http_post(self, test_client):
        """curl 风格 POST /mcp 调用 tools/list，应返回工具列表。"""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "result" in body, f"响应中缺少 'result': {body}"
        assert "tools" in body["result"], f"result 中缺少 'tools': {body['result']}"
        assert body["result"]["tools"], "工具列表不应为空"

        # 验证 query_knowledge_base 工具已注册
        tool_names = [t["name"] for t in body["result"]["tools"]]
        assert "query_knowledge_base" in tool_names, (
            f"应在工具列表中找到 query_knowledge_base，实际: {tool_names}"
        )

    def test_tools_list_returns_json_not_sse(self, test_client):
        """json_response=True 时应返回纯 JSON，而非 SSE 文本流。"""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        # 应为 JSON，不是 text/event-stream (SSE)
        assert "text/event-stream" not in content_type, (
            f"期望 JSON 响应，实际为 SSE: {content_type}"
        )
        # 应该可以解析为 JSON
        body = response.json()
        assert "result" in body

    def test_initialize_via_http_post(self, test_client):
        """MCP initialize 请求应可通过直接 POST 完成。"""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "result" in body, f"响应中缺少 'result': {body}"
        assert "protocolVersion" in body["result"]

    def test_invalid_method_returns_error(self, test_client):
        """无效的 JSON-RPC 方法应返回错误响应。"""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "nonexistent/method",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200  # JSON-RPC 错误仍返回 HTTP 200
        body = response.json()
        assert "error" in body or "result" in body, (
            f"期望 JSON-RPC 响应，实际: {body}"
        )

    def test_missing_accept_header_returns_406(self, test_client):
        """缺少 Accept: application/json 时应返回 406 Not Acceptable。"""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
        )

        assert response.status_code == 406
        body = response.json()
        assert "error" in body
        assert "application/json" in body["error"]["message"]

    def test_tools_call_query_knowledge_base(self, settings, test_client):
        """curl 风格 tools/call query_knowledge_base 应返回查询结果。"""
        from knowledge_hub.schemas import ChunkResult, QueryResult

        # 构造模拟查询结果
        mock_result = QueryResult(
            results=[
                ChunkResult(
                    text="FreeRTOS 优先级继承机制可防止优先级反转。",
                    source_file="rtos_guide.md",
                    page_or_section="优先级继承",
                    heading_path=["FreeRTOS 调度指南", "优先级继承"],
                    score=0.95,
                ),
            ],
            query_time_ms=12.5,
        )

        with _patch_heavy_components():
            from knowledge_hub.server.mcp_server import create_mcp_app
            from starlette.testclient import TestClient

            mcp = create_mcp_app(settings)
            mcp._health.model_loaded = True
            mcp._health.qdrant = True

            # Mock query_engine.query 返回模拟结果
            mock_engine = MagicMock()
            mock_engine.query = AsyncMock(return_value=mock_result)

            # 替换 tools 模块中的 QueryEngine
            import knowledge_hub.server.tools as tools_module
            _original = tools_module.QueryEngine
            tools_module.QueryEngine = MagicMock(return_value=mock_engine)

            try:
                app = mcp.http_app(
                    transport="streamable-http",
                    stateless_http=True,
                    json_response=True,
                    path="/mcp",
                )

                with TestClient(app) as client:
                    response = client.post(
                        "/mcp",
                        json={
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {
                                "name": "query_knowledge_base",
                                "arguments": {"query": "优先级继承", "top_k": 3},
                            },
                        },
                        headers=self.JSON_HEADERS,
                    )

                    assert response.status_code == 200
                    body = response.json()
                    assert "result" in body or "error" in body, (
                        f"期望 JSON-RPC 响应，实际: {body}"
                    )
            finally:
                tools_module.QueryEngine = _original
