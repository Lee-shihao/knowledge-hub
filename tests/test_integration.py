"""End-to-end integration test: ingest a markdown file and query it.

Requires:
- Qdrant running on localhost:6333 (e.g., via Docker)
- FlagEmbedding models will be loaded on CPU (first run downloads ~2.2GB)

Run with:
    pytest tests/test_integration.py -v -s

Skip integration tests with:
    pytest -m "not integration"
"""
import pytest
from pathlib import Path

from qdrant_client import QdrantClient

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.ingestion.pipeline import IngestionPipeline
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput


def qdrant_available() -> bool:
    """Check if Qdrant is reachable on localhost:6333."""
    try:
        client = QdrantClient("http://localhost:6333", check_compatibility=False)
        client.get_collections()
        return True
    except Exception:
        return False


# Skip entire module if Qdrant is not available
pytestmark = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available on localhost:6333 — start with: docker run -p 6333:6333 qdrant/qdrant",
)


@pytest.fixture
def settings(temp_storage_dir, tmp_path):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_integration",
        EMBED_DEVICE="cpu",
        STORAGE_DIR=str(temp_storage_dir),
        DATA_DIR=str(tmp_path / "data"),
    )


@pytest.fixture
async def integration_setup(settings):
    """Set up all components for integration testing with real Qdrant + FlagEmbedding."""
    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()

    # Real FlagEmbedding on CPU
    embedder = FlagEmbeddingEmbedder(settings)
    reranker = Reranker(settings)

    pipeline = IngestionPipeline(
        settings=settings,
        loader=DocumentLoader(settings),
        chunker=SemanticChunker(settings),
        embedder=embedder,
        vector_store=store,
        metadata_mgr=meta_mgr,
    )

    engine = QueryEngine(settings, embedder, store, reranker)

    yield {
        "pipeline": pipeline,
        "engine": engine,
        "store": store,
        "meta_mgr": meta_mgr,
        "client": client,
    }

    # Cleanup
    try:
        client.delete_collection(settings.QDRANT_COLLECTION)
    except Exception:
        pass
    try:
        client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")
    except Exception:
        pass


@pytest.fixture
def rtdoc_path(tmp_path):
    """Create a test RTOS document."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc_path = data_dir / "rtos_guide.md"
    doc_path.write_text("""# FreeRTOS Scheduling Guide

## Priority Inheritance

Priority inheritance is a mechanism that prevents priority inversion.
When a high-priority task is blocked by a low-priority task holding a mutex,
the low-priority task temporarily inherits the high priority until it
releases the mutex. This ensures the high-priority task doesn't get
indefinitely blocked by medium-priority tasks.

## Task Notifications

Task notifications are a lightweight alternative to semaphores in FreeRTOS.
Each task has a 32-bit notification value that can be updated by other tasks
or interrupts. Notifications are faster and use less RAM than semaphores.
""")
    return doc_path


@pytest.mark.asyncio
async def test_full_ingest_and_query(integration_setup, rtdoc_path, settings):
    """Ingest a markdown document and verify it can be queried with real embeddings."""
    pipeline = integration_setup["pipeline"]
    engine = integration_setup["engine"]
    store = integration_setup["store"]

    # Ingest
    report = await pipeline.run([rtdoc_path], tags=["rtos", "freertos"])
    assert report.succeeded == 1
    assert report.failed == 0
    assert report.skipped == 0

    # Verify chunks were stored
    count = await store.count()
    assert count > 0

    # Query
    result = await engine.query(QueryInput(query="priority inheritance mutex", top_k=2))
    assert len(result.results) > 0
    assert result.query_time_ms > 0
    # The result should contain relevant content
    found = any("inheritance" in r.text.lower() for r in result.results)
    assert found, f"Expected 'inheritance' in results, got: {[r.text[:100] for r in result.results]}"


@pytest.mark.asyncio
async def test_ingest_skips_unchanged_file(integration_setup, rtdoc_path, settings):
    """Re-ingesting the same file should skip it."""
    pipeline = integration_setup["pipeline"]

    # First ingestion
    report1 = await pipeline.run([rtdoc_path])
    assert report1.succeeded == 1
    assert report1.skipped == 0

    # Second ingestion — should skip
    report2 = await pipeline.run([rtdoc_path])
    assert report2.skipped == 1
    assert report2.succeeded == 0


@pytest.mark.asyncio
async def test_ingest_reingests_changed_file(integration_setup, tmp_path, settings):
    """Changed file should be re-ingested."""
    pipeline = integration_setup["pipeline"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc_path = data_dir / "doc.md"

    # Write initial version
    doc_path.write_text("# Version 1\n\nOriginal content here. " * 10)
    report1 = await pipeline.run([doc_path])
    assert report1.succeeded == 1

    # Modify the file
    doc_path.write_text("# Version 2\n\nUpdated content here. " * 10)
    report2 = await pipeline.run([doc_path])
    assert report2.succeeded == 1
    assert report2.skipped == 0


@pytest.mark.asyncio
async def test_orphan_cleanup(integration_setup, rtdoc_path, settings):
    """Orphan cleanup should remove vectors for deleted files."""
    pipeline = integration_setup["pipeline"]
    store = integration_setup["store"]

    # Ingest
    report = await pipeline.run([rtdoc_path])
    assert report.succeeded == 1
    count_after = await store.count()
    assert count_after > 0

    # Delete the file and run with empty paths
    rtdoc_path.unlink()
    report2 = await pipeline.run([])
    assert report2.orphans_cleaned == 1


@pytest.mark.asyncio
async def test_query_with_source_filter(integration_setup, rtdoc_path, settings):
    """Query with source filter should only return matching results."""
    pipeline = integration_setup["pipeline"]
    engine = integration_setup["engine"]

    # Ingest
    await pipeline.run([rtdoc_path])

    # Query with matching source filter
    result = await engine.query(
        QueryInput(query="priority", top_k=5, filter_source="rtos_guide.md")
    )
    assert len(result.results) > 0

    # Query with non-matching source filter
    result2 = await engine.query(
        QueryInput(query="priority", top_k=5, filter_source="nonexistent.md")
    )
    assert len(result2.results) == 0


@pytest.mark.asyncio
async def test_multiple_file_ingestion(integration_setup, tmp_path, settings):
    """Ingesting multiple files should produce chunks from all files."""
    pipeline = integration_setup["pipeline"]
    store = integration_setup["store"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create two documents
    (data_dir / "doc1.md").write_text("# Document 1\n\nContent for doc one. " * 10)
    (data_dir / "doc2.md").write_text("# Document 2\n\nContent for doc two. " * 10)

    report = await pipeline.run([data_dir / "doc1.md", data_dir / "doc2.md"])
    assert report.succeeded == 2
    assert report.failed == 0


@pytest.mark.asyncio
async def test_query_empty_collection(integration_setup, settings):
    """Querying an empty collection should return empty results."""
    engine = integration_setup["engine"]

    result = await engine.query(QueryInput(query="anything", top_k=5))
    assert len(result.results) == 0
    assert result.query_time_ms >= 0
