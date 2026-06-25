"""Tests for MCP tools — query_knowledge_base health gates and QueryEngine calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from knowledge_hub.config import Settings
from knowledge_hub.server.tools import create_tools


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def mock_health():
    h = MagicMock()
    h.get_status = AsyncMock()
    return h


@pytest.fixture
def mock_query_engine():
    e = MagicMock()
    e.query = AsyncMock()
    return e


@pytest.fixture
def mock_metadata_mgr():
    m = MagicMock()
    m.list_source_details = AsyncMock(return_value=[])
    m.list_sources = AsyncMock(return_value=[])
    return m


@pytest.fixture
def mock_vector_store():
    v = MagicMock()
    v.count = AsyncMock(return_value=0)
    return v


class TestCreateTools:
    """Tests for create_tools() — tool registration and count."""

    def test_returns_three_tools(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """create_tools() should return exactly 3 tools."""
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        assert len(tools) == 3
        assert set(tools.keys()) == {
            "query_knowledge_base",
            "list_kb_sources",
            "get_kb_status",
        }

    def test_tools_are_async_callables(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """All returned tools should be async callables."""
        import inspect
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        for name, fn in tools.items():
            assert inspect.iscoroutinefunction(fn), (
                f"{name} should be a coroutine function"
            )


class TestQueryKnowledgeBase:
    """Tests for query_knowledge_base tool."""

    @pytest.mark.asyncio
    async def test_returns_error_when_model_not_loaded(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return error dict when model_loaded is False."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=False, qdrant=True,
        )
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["query_knowledge_base"]("test query")
        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_calls_query_engine_when_healthy(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should call query_engine.query() when health checks pass."""
        from knowledge_hub.schemas import QueryResult, ChunkResult

        mock_health.get_status.return_value = MagicMock(
            model_loaded=True, qdrant=True,
        )
        mock_result = QueryResult(
            results=[
                ChunkResult(
                    text="test text",
                    source_file="test.md",
                    page_or_section="section 1",
                    heading_path=["H1"],
                    score=0.95,
                ),
            ],
            query_time_ms=10.0,
        )
        mock_query_engine.query.return_value = mock_result

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["query_knowledge_base"]("test query", top_k=3)
        assert "error" not in result
        assert result["results"][0]["text"] == "test text"
        mock_query_engine.query.assert_called_once()


class TestListKbSources:
    """Tests for list_kb_sources tool."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_sources(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return empty sources list when nothing is indexed."""
        mock_metadata_mgr.list_source_details.return_value = []
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["list_kb_sources"]()
        assert result["sources"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_returns_sources_with_metadata(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return source details including chunk_count and hash."""
        mock_metadata_mgr.list_source_details.return_value = [
            {"source_file": "doc.md", "source_hash": "abc", "chunk_count": 5},
            {"source_file": "doc.pdf", "source_hash": "def", "chunk_count": 3},
        ]
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["list_kb_sources"]()
        assert result["count"] == 2
        assert result["sources"][0]["filename"] == "doc.md"
        assert result["sources"][0]["chunk_count"] == 5
        assert result["sources"][0]["source_hash"] == "abc"


class TestGetKbStatus:
    """Tests for get_kb_status tool."""

    @pytest.mark.asyncio
    async def test_returns_health_and_stats(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return health status plus collection statistics."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=True,
            qdrant=True,
            gpu_available=True,
            gpu_memory_free_mb=8192,
        )
        mock_vector_store.count.return_value = 42
        mock_metadata_mgr.list_sources.return_value = ["a.md", "b.md"]

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["get_kb_status"]()
        assert result["model_loaded"] is True
        assert result["qdrant"] is True
        assert result["total_chunks"] == 42
        assert result["total_sources"] == 2
        assert result["collection"] == "knowledge_hub"

    @pytest.mark.asyncio
    async def test_handles_vector_store_error(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return -1 counts when Qdrant is unreachable."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=True, qdrant=False,
            gpu_available=False, gpu_memory_free_mb=0,
        )
        mock_vector_store.count.side_effect = RuntimeError("qdrant down")
        mock_metadata_mgr.list_sources.side_effect = RuntimeError("qdrant down")

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["get_kb_status"]()
        assert result["total_chunks"] == -1
        assert result["total_sources"] == -1
