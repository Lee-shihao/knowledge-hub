"""Tests for MCP server creation — app wiring, auth, IP middleware."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_hub.config import Settings


def _make_app_state(settings, **overrides):
    """Build a mock AppState with the given settings and overridable fields."""
    from knowledge_hub.server.app_state import AppState

    defaults = {
        "settings": settings,
        "qdrant_client": MagicMock(),
        "embedder": MagicMock(),
        "reranker": MagicMock(),
        "metadata_mgr": MagicMock(),
        "vector_store": MagicMock(),
        "query_engine": MagicMock(),
        "pipeline": MagicMock(),
        "health": MagicMock(),
        "job_manager": MagicMock(),
    }
    defaults.update(overrides)
    return AppState(**defaults)


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
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        from fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_has_health_attribute(self, settings):
        """create_mcp_app should attach _health from AppState."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        assert hasattr(mcp, "_health")
        assert mcp._health is state.health

    def test_create_mcp_app_with_auth_token(self, settings_with_auth):
        """When MCP_AUTH_TOKEN is set, auth should be configured without error."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_with_auth)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_raises_on_lan_no_auth(self, settings_lan_no_auth):
        """When MCP_HOST is not 127.0.0.1 and no auth token, should raise ValueError."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_lan_no_auth)
        with pytest.raises(ValueError, match="MCP_HOST must be 127.0.0.1"):
            create_mcp_app(state)

    def test_create_mcp_app_allows_localhost_no_auth(self, settings):
        """When MCP_HOST is 127.0.0.1 and no auth token, should succeed."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_with_allowed_ips(self, settings_with_allowed_ips):
        """When MCP_ALLOWED_IPS is set, IP middleware should be added without error."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_with_allowed_ips)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"


class TestStreamableHTTPIntegration:
    """curl-style integration tests — verify streamable-http transport via TestClient."""

    JSON_HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def test_client(self, settings):
        """Create MCP app via AppState and return Starlette TestClient."""
        from starlette.testclient import TestClient
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
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
        """curl-style POST /mcp calling tools/list should return tool list."""
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
        assert "result" in body, f"Missing 'result' in: {body}"
        assert "tools" in body["result"]
        assert body["result"]["tools"], "Tool list should not be empty"

        tool_names = [t["name"] for t in body["result"]["tools"]]
        assert "query_knowledge_base" in tool_names, (
            f"Expected query_knowledge_base in tools, got: {tool_names}"
        )
        assert "list_kb_sources" in tool_names, (
            f"Expected list_kb_sources in tools, got: {tool_names}"
        )
        assert "get_kb_status" in tool_names, (
            f"Expected get_kb_status in tools, got: {tool_names}"
        )

    def test_tools_list_returns_json_not_sse(self, test_client):
        """json_response=True should return JSON, not SSE text stream."""
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
        assert "text/event-stream" not in content_type
        body = response.json()
        assert "result" in body

    def test_initialize_via_http_post(self, test_client):
        """MCP initialize request should succeed via direct POST."""
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
        assert "result" in body
        assert "protocolVersion" in body["result"]

    def test_invalid_method_returns_error(self, test_client):
        """Invalid JSON-RPC method should return error response."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "nonexistent/method",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "error" in body or "result" in body

    def test_missing_accept_header_returns_406(self, test_client):
        """Missing Accept: application/json should return 406."""
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
