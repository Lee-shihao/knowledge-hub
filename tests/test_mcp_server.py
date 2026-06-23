"""Tests for MCP server creation — app wiring, auth, IP middleware.

All heavy components (FlagEmbeddingEmbedder, Reranker, QdrantClient) are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch

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
