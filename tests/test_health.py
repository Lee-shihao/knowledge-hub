"""Tests for HealthMonitor (no real Qdrant/GPU probing, mock-based only)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from knowledge_hub.config import Settings
from knowledge_hub.server.health import HealthStatus, HealthMonitor


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_health_status_defaults(self):
        """HealthStatus should have correct defaults."""
        status = HealthStatus()
        assert status.model_loaded is False
        assert status.qdrant is False
        assert status.gpu_available is False
        assert status.gpu_memory_free_mb == 0

    def test_health_status_custom_values(self):
        """HealthStatus should accept custom values."""
        status = HealthStatus(
            model_loaded=True,
            qdrant=True,
            gpu_available=True,
            gpu_memory_free_mb=8000,
        )
        assert status.model_loaded is True
        assert status.qdrant is True
        assert status.gpu_available is True
        assert status.gpu_memory_free_mb == 8000


class TestHealthMonitor:
    """Tests for HealthMonitor (mocked)."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def mock_qdrant(self):
        """Mock QdrantClient for testing."""
        client = MagicMock()
        client.get_collections = MagicMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_probe_qdrant_returns_true_on_success(self, settings, mock_qdrant):
        """_probe_qdrant should return True when Qdrant responds."""
        monitor = HealthMonitor(settings, mock_qdrant)
        result = await monitor._probe_qdrant()
        assert result is True
        mock_qdrant.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_probe_qdrant_returns_false_on_failure(self, settings, mock_qdrant):
        """_probe_qdrant should return False on exception."""
        mock_qdrant.get_collections.side_effect = Exception("Connection refused")
        monitor = HealthMonitor(settings, mock_qdrant)
        result = await monitor._probe_qdrant()
        assert result is False

    @pytest.mark.asyncio
    async def test_probe_gpu_returns_tuple(self, settings, mock_qdrant):
        """_probe_gpu should return (bool, int) tuple."""
        monitor = HealthMonitor(settings, mock_qdrant)
        result = await monitor._probe_gpu()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    @pytest.mark.asyncio
    async def test_probe_all_returns_health_status(self, settings, mock_qdrant):
        """_probe_all should return a HealthStatus instance."""
        monitor = HealthMonitor(settings, mock_qdrant)
        status = await monitor._probe_all()
        assert isinstance(status, HealthStatus)
        # model_loaded is always True (server started = models loaded)
        assert status.model_loaded is True

    @pytest.mark.asyncio
    async def test_get_status_returns_cached_status(self, settings, mock_qdrant):
        """get_status should return cached status without re-probing."""
        monitor = HealthMonitor(settings, mock_qdrant)
        status1 = await monitor.get_status()
        status2 = await monitor.get_status()
        # Both calls should return the same cached object
        assert status1 is status2

    @pytest.mark.asyncio
    async def test_start_initializes_cache(self, settings, mock_qdrant):
        """start() should initialize the cached status."""
        monitor = HealthMonitor(settings, mock_qdrant)
        assert monitor._cached_status is None
        await monitor.start(interval_seconds=30)
        assert monitor._cached_status is not None
        assert isinstance(monitor._cached_status, HealthStatus)
        # Cancel the probe loop task
        if monitor._task:
            monitor._task.cancel()
            try:
                await monitor._task
            except asyncio.CancelledError:
                pass