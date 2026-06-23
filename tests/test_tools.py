"""Tests for MCP tools — query_knowledge_base health gates and QueryEngine calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from knowledge_hub.config import Settings
from knowledge_hub.schemas import QueryInput, QueryResult, ChunkResult
from knowledge_hub.server.health import HealthStatus, HealthMonitor
from knowledge_hub.server.tools import create_tools


class TestCreateTools:
    """Tests for create_tools() and the query_knowledge_base function."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def mock_query_engine(self):
        """Mock QueryEngine with query() returning a QueryResult."""
        engine = MagicMock()
        engine.query = AsyncMock(return_value=QueryResult(
            results=[
                ChunkResult(
                    text="Sample result text",
                    source_file="test.md",
                    page_or_section="Section 1",
                    heading_path=["Section 1"],
                    score=0.95,
                ),
            ],
            query_time_ms=12.5,
        ))
        return engine

    @pytest.fixture
    def mock_health(self):
        """Mock HealthMonitor with healthy status."""
        health = MagicMock(spec=HealthMonitor)
        health.get_status = AsyncMock(return_value=HealthStatus(
            model_loaded=True,
            qdrant=True,
            gpu_available=True,
            gpu_memory_free_mb=8000,
        ))
        return health

    @pytest.fixture
    def mock_health_unhealthy_model(self):
        """Mock HealthMonitor with model not loaded."""
        health = MagicMock(spec=HealthMonitor)
        health.get_status = AsyncMock(return_value=HealthStatus(
            model_loaded=False,
            qdrant=True,
            gpu_available=False,
            gpu_memory_free_mb=0,
        ))
        return health

    @pytest.fixture
    def mock_health_unhealthy_qdrant(self):
        """Mock HealthMonitor with Qdrant down."""
        health = MagicMock(spec=HealthMonitor)
        health.get_status = AsyncMock(return_value=HealthStatus(
            model_loaded=True,
            qdrant=False,
            gpu_available=True,
            gpu_memory_free_mb=8000,
        ))
        return health

    def test_create_tools_returns_dict_with_query_knowledge_base(self, settings, mock_query_engine, mock_health):
        """create_tools should return a dict with 'query_knowledge_base' key."""
        tools = create_tools(settings, mock_query_engine, mock_health)
        assert "query_knowledge_base" in tools
        assert callable(tools["query_knowledge_base"])

    @pytest.mark.asyncio
    async def test_query_knowledge_base_calls_query_engine(self, settings, mock_query_engine, mock_health):
        """query_knowledge_base should call QueryEngine.query with correct params."""
        tools = create_tools(settings, mock_query_engine, mock_health)
        fn = tools["query_knowledge_base"]

        result = await fn(query="test query", top_k=3)

        # Verify the engine was called
        mock_query_engine.query.assert_called_once()
        call_arg = mock_query_engine.query.call_args[0][0]
        assert isinstance(call_arg, QueryInput)
        assert call_arg.query == "test query"
        assert call_arg.top_k == 3
        assert call_arg.filter_source is None
        assert call_arg.filter_tags is None

        # Verify result structure
        assert isinstance(result, dict)
        assert "results" in result
        assert "query_time_ms" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["source_file"] == "test.md"

    @pytest.mark.asyncio
    async def test_query_knowledge_base_passes_filters(self, settings, mock_query_engine, mock_health):
        """query_knowledge_base should pass filter_source and filter_tags."""
        tools = create_tools(settings, mock_query_engine, mock_health)
        fn = tools["query_knowledge_base"]

        await fn(
            query="filtered query",
            top_k=10,
            filter_source="data/test.pdf",
            filter_tags=["python", "api"],
        )

        call_arg = mock_query_engine.query.call_args[0][0]
        assert call_arg.filter_source == "data/test.pdf"
        assert call_arg.filter_tags == ["python", "api"]

    @pytest.mark.asyncio
    async def test_query_knowledge_base_blocks_on_model_not_loaded(self, settings, mock_query_engine, mock_health_unhealthy_model):
        """When model_loaded=False, should return error dict without calling engine."""
        tools = create_tools(settings, mock_query_engine, mock_health_unhealthy_model)
        fn = tools["query_knowledge_base"]

        result = await fn(query="test query")

        assert result == {
            "error": "Embedding model is not available",
            "results": [],
            "query_time_ms": 0,
        }
        mock_query_engine.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_knowledge_base_blocks_on_qdrant_down(self, settings, mock_query_engine, mock_health_unhealthy_qdrant):
        """When qdrant=False, should return error dict without calling engine."""
        tools = create_tools(settings, mock_query_engine, mock_health_unhealthy_qdrant)
        fn = tools["query_knowledge_base"]

        result = await fn(query="test query")

        assert result == {
            "error": "Knowledge base is not available",
            "results": [],
            "query_time_ms": 0,
        }
        mock_query_engine.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_knowledge_base_default_top_k(self, settings, mock_query_engine, mock_health):
        """query_knowledge_base should use default top_k=5."""
        tools = create_tools(settings, mock_query_engine, mock_health)
        fn = tools["query_knowledge_base"]

        await fn(query="test")
        call_arg = mock_query_engine.query.call_args[0][0]
        assert call_arg.top_k == 5
