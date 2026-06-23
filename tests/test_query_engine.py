"""Tests for QueryEngine — orchestrates embed → hybrid search → rerank."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from knowledge_hub.config import Settings
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput, QueryResult, ChunkResult


@pytest.fixture
def settings(temp_storage_dir):
    """Settings with test defaults."""
    return Settings(STORAGE_DIR=str(temp_storage_dir))


@pytest.fixture
def embedder():
    """Mock FlagEmbeddingEmbedder."""
    mock = AsyncMock()
    mock.embed_query.return_value = {
        "dense": [0.1] * 1024,
        "sparse": {1: 0.5, 2: 0.3},
    }
    return mock


@pytest.fixture
def vector_store():
    """Mock QdrantVectorStore."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def reranker():
    """Mock Reranker."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def query_engine(settings, embedder, vector_store, reranker):
    """Create QueryEngine with all mocked dependencies."""
    return QueryEngine(settings, embedder, vector_store, reranker)


# ---- Test 1: Empty collection returns empty results ----

@pytest.mark.asyncio
async def test_query_empty_collection(query_engine, vector_store):
    """Query with no candidates from hybrid search returns empty QueryResult."""
    vector_store.hybrid_search.return_value = []

    result = await query_engine.query(QueryInput(query="test"))

    assert isinstance(result, QueryResult)
    assert len(result.results) == 0
    assert result.query_time_ms >= 0


# ---- Test 2: Query with results produces correct ChunkResult ----

@pytest.mark.asyncio
async def test_query_with_results(query_engine, vector_store, reranker):
    """Query with candidates produces correct ChunkResult items."""
    vector_store.hybrid_search.return_value = [
        ("chunk_1", 0.9, {
            "text": "Priority inheritance prevents inversion",
            "source_file": "rtos.md",
            "source_hash": "abc123",
            "page_number": 5,
            "heading_path": ["Introduction", "Scheduling"],
            "tags": ["rtos"],
        }),
    ]

    reranker.rerank.return_value = [
        {
            "text": "Priority inheritance prevents inversion",
            "score": 0.95,
            "source_file": "rtos.md",
            "source_hash": "abc123",
            "page_number": 5,
            "heading_path": ["Introduction", "Scheduling"],
            "tags": ["rtos"],
        },
    ]

    result = await query_engine.query(QueryInput(query="priority inheritance"))

    assert len(result.results) == 1
    chunk = result.results[0]
    assert isinstance(chunk, ChunkResult)
    assert chunk.text == "Priority inheritance prevents inversion"
    assert chunk.source_file == "rtos.md"
    assert chunk.page_or_section == "Scheduling"  # heading_path[-1]
    assert chunk.heading_path == ["Introduction", "Scheduling"]
    assert chunk.score == 0.95
    assert result.query_time_ms >= 0


# ---- Test 3: filter_source and filter_tags passed to hybrid_search ----

@pytest.mark.asyncio
async def test_query_passes_filters_to_hybrid_search(query_engine, vector_store, reranker):
    """Query passes filter_source and filter_tags to hybrid_search."""
    vector_store.hybrid_search.return_value = []
    reranker.rerank.return_value = []

    q = QueryInput(query="test", filter_source="doc.md", filter_tags=["tag1", "tag2"])
    await query_engine.query(q)

    vector_store.hybrid_search.assert_called_once()
    call_kwargs = vector_store.hybrid_search.call_args
    assert call_kwargs.kwargs["filter_source"] == "doc.md"
    assert call_kwargs.kwargs["filter_tags"] == ["tag1", "tag2"]


# ---- Test 4: HYBRID_CANDIDATE_K for hybrid search, top_k for rerank ----

@pytest.mark.asyncio
async def test_query_uses_correct_k_values(query_engine, vector_store, reranker, settings):
    """Hybrid search uses HYBRID_CANDIDATE_K; rerank uses top_k from QueryInput."""
    vector_store.hybrid_search.return_value = [
        ("c1", 0.9, {"text": "a", "source_file": "f.md", "page_number": 1, "heading_path": [], "tags": []}),
    ]
    reranker.rerank.return_value = []

    q = QueryInput(query="test", top_k=3)
    await query_engine.query(q)

    # hybrid_search should be called with HYBRID_CANDIDATE_K
    hybrid_call = vector_store.hybrid_search.call_args
    assert hybrid_call.kwargs["top_k"] == settings.HYBRID_CANDIDATE_K

    # rerank should be called with top_k from QueryInput
    rerank_call = reranker.rerank.call_args
    assert rerank_call.kwargs["top_k"] == 3


# ---- Test 5: Reranker failure → graceful degradation ----

@pytest.mark.asyncio
async def test_query_reranker_failure_graceful_degradation(query_engine, vector_store, reranker):
    """When reranker fails, results still returned (graceful degradation)."""
    vector_store.hybrid_search.return_value = [
        ("c1", 0.9, {
            "text": "Document text",
            "source_file": "doc.md",
            "source_hash": "h1",
            "page_number": 1,
            "heading_path": ["Intro"],
            "tags": [],
        }),
    ]

    # Reranker returns original candidates on failure (graceful degradation)
    reranker.rerank.return_value = [
        {
            "text": "Document text",
            "score": 0.9,
            "source_file": "doc.md",
            "source_hash": "h1",
            "page_number": 1,
            "heading_path": ["Intro"],
            "tags": [],
        },
    ]

    result = await query_engine.query(QueryInput(query="test"))

    assert len(result.results) == 1
    assert result.results[0].text == "Document text"


# ---- Test 6: page_or_section construction ----

@pytest.mark.asyncio
async def test_page_or_section_from_heading_path(query_engine, vector_store, reranker):
    """page_or_section uses heading_path[-1] when available."""
    vector_store.hybrid_search.return_value = [
        ("c1", 0.8, {
            "text": "text",
            "source_file": "doc.md",
            "page_number": 3,
            "heading_path": ["Chapter 1", "Section A"],
            "tags": [],
        }),
    ]
    reranker.rerank.return_value = [
        {
            "text": "text",
            "score": 0.8,
            "source_file": "doc.md",
            "page_number": 3,
            "heading_path": ["Chapter 1", "Section A"],
            "tags": [],
        },
    ]

    result = await query_engine.query(QueryInput(query="test"))
    assert result.results[0].page_or_section == "Section A"


@pytest.mark.asyncio
async def test_page_or_section_from_page_number(query_engine, vector_store, reranker):
    """page_or_section uses f'p{page_number}' when heading_path is empty."""
    vector_store.hybrid_search.return_value = [
        ("c1", 0.7, {
            "text": "text",
            "source_file": "doc.md",
            "page_number": 7,
            "heading_path": [],
            "tags": [],
        }),
    ]
    reranker.rerank.return_value = [
        {
            "text": "text",
            "score": 0.7,
            "source_file": "doc.md",
            "page_number": 7,
            "heading_path": [],
            "tags": [],
        },
    ]

    result = await query_engine.query(QueryInput(query="test"))
    assert result.results[0].page_or_section == "p7"


@pytest.mark.asyncio
async def test_page_or_section_empty_when_no_heading_or_page(query_engine, vector_store, reranker):
    """page_or_section is empty string when no heading_path or page_number."""
    vector_store.hybrid_search.return_value = [
        ("c1", 0.6, {
            "text": "text",
            "source_file": "doc.md",
            "page_number": None,
            "heading_path": [],
            "tags": [],
        }),
    ]
    reranker.rerank.return_value = [
        {
            "text": "text",
            "score": 0.6,
            "source_file": "doc.md",
            "page_number": None,
            "heading_path": [],
            "tags": [],
        },
    ]

    result = await query_engine.query(QueryInput(query="test"))
    assert result.results[0].page_or_section == ""


# ---- Test 7: Query calls embed_query with correct text ----

@pytest.mark.asyncio
async def test_query_calls_embed_query(query_engine, embedder, vector_store):
    """Query calls embedder.embed_query with the query string."""
    vector_store.hybrid_search.return_value = []

    await query_engine.query(QueryInput(query="hello world"))

    embedder.embed_query.assert_called_once_with("hello world")


# ---- Test 8: Query passes dense and sparse vectors to hybrid_search ----

@pytest.mark.asyncio
async def test_query_passes_vectors_to_hybrid_search(query_engine, embedder, vector_store):
    """Query passes dense and sparse vectors from embedder to hybrid_search."""
    embedder.embed_query.return_value = {
        "dense": [0.2] * 1024,
        "sparse": {10: 0.8, 20: 0.4},
    }
    vector_store.hybrid_search.return_value = []

    await query_engine.query(QueryInput(query="test"))

    call_kwargs = vector_store.hybrid_search.call_args.kwargs
    assert call_kwargs["dense_vec"] == [0.2] * 1024
    assert call_kwargs["sparse_vec"] == {10: 0.8, 20: 0.4}
