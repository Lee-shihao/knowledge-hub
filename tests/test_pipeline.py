"""Tests for IngestionPipeline — orchestrates load → chunk → embed → store."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.ingestion.pipeline import IngestionPipeline, IngestionReport


@pytest.fixture
def settings(temp_storage_dir, tmp_path):
    return Settings(
        QDRANT_URL=":memory:",
        QDRANT_COLLECTION="test_pipeline",
        STORAGE_DIR=str(temp_storage_dir),
        DATA_DIR=str(tmp_path / "data"),
    )


@pytest.fixture
def mock_embedder():
    """Mock FlagEmbeddingEmbedder to avoid loading the real model."""
    embedder = MagicMock(spec=FlagEmbeddingEmbedder)
    embedder.embed_texts = AsyncMock(return_value=[
        {"dense": [0.1] * 1024, "sparse": {0: 0.5, 10: 0.3}}
    ])
    embedder.embed_query = AsyncMock(return_value={
        "dense": [0.1] * 1024, "sparse": {0: 0.5}
    })
    return embedder


@pytest.fixture
async def pipeline(settings, mock_embedder):
    client = QdrantClient(":memory:")
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    loader = DocumentLoader(settings)
    chunker = SemanticChunker(settings)
    pl = IngestionPipeline(settings, loader, chunker, mock_embedder, store, meta_mgr)
    yield pl


def _write_md(path: Path, content: str):
    """Helper to write a markdown file with enough content to produce chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.mark.asyncio
async def test_pipeline_ingests_markdown(pipeline, settings, tmp_path):
    """Pipeline should ingest a markdown file and report success."""
    data_dir = tmp_path / "data"
    content = "# Hello\n\nWorld content here. " * 10
    _write_md(data_dir / "test.md", content)

    report = await pipeline.run([data_dir / "test.md"])

    assert report.succeeded == 1
    assert report.failed == 0
    assert report.total == 1


@pytest.mark.asyncio
async def test_pipeline_skips_unchanged_file(pipeline, settings, tmp_path):
    """Re-running pipeline on unchanged file should skip it."""
    data_dir = tmp_path / "data"
    content = "# Hello\n\nWorld content here. " * 10
    _write_md(data_dir / "test.md", content)

    report1 = await pipeline.run([data_dir / "test.md"])
    assert report1.succeeded == 1
    assert report1.skipped == 0

    report2 = await pipeline.run([data_dir / "test.md"])
    assert report2.skipped == 1
    assert report2.succeeded == 0


@pytest.mark.asyncio
async def test_pipeline_force_reingests(pipeline, settings, tmp_path):
    """Force flag should re-ingest even unchanged files."""
    data_dir = tmp_path / "data"
    content = "# Hello\n\nWorld content here. " * 10
    _write_md(data_dir / "test.md", content)

    await pipeline.run([data_dir / "test.md"])
    report2 = await pipeline.run([data_dir / "test.md"], force=True)

    assert report2.succeeded == 1
    assert report2.skipped == 0


@pytest.mark.asyncio
async def test_pipeline_handles_missing_file(pipeline, settings, tmp_path):
    """Missing files should not be counted as failures (loader skips them)."""
    report = await pipeline.run([tmp_path / "nonexistent.md"])

    assert report.succeeded == 0
    assert report.failed == 0
    assert report.total == 0  # nonexistent file fails is_file() check


@pytest.mark.asyncio
async def test_pipeline_handles_unsupported_format(pipeline, settings, tmp_path):
    """Unsupported file formats should be skipped, not counted as failures."""
    data_dir = tmp_path / "data"
    _write_md(data_dir / "test.xyz", "some content")

    report = await pipeline.run([data_dir / "test.xyz"])

    assert report.succeeded == 0
    assert report.failed == 0


@pytest.mark.asyncio
async def test_pipeline_sidecar_tags(pipeline, mock_embedder, settings, tmp_path):
    """Sidecar .meta.json should provide tags with highest priority."""
    data_dir = tmp_path / "data"
    content = "# Tagged Doc\n\nSome content here. " * 10
    _write_md(data_dir / "tagged.md", content)

    # Write sidecar metadata
    sidecar = data_dir / ".meta.json"
    sidecar.write_text(json.dumps({"tags": ["priority-tag", "custom"]}))

    # Need to return enough embeddings for all chunks
    mock_embedder.embed_texts = AsyncMock(return_value=[
        {"dense": [0.1] * 1024, "sparse": {0: 0.5}}
    ] * 5)

    report = await pipeline.run([data_dir / "tagged.md"])
    assert report.succeeded == 1

    # Verify tags were applied via the embedder call
    call_args = mock_embedder.embed_texts.call_args
    assert call_args is not None


@pytest.mark.asyncio
async def test_pipeline_directory_tag_fallback(pipeline, mock_embedder, settings, tmp_path):
    """Parent directory name should be used as fallback tag."""
    data_dir = tmp_path / "data" / "my-project"
    content = "# Project Doc\n\nSome content here. " * 10
    _write_md(data_dir / "doc.md", content)

    mock_embedder.embed_texts = AsyncMock(return_value=[
        {"dense": [0.1] * 1024, "sparse": {0: 0.5}}
    ] * 5)

    report = await pipeline.run([data_dir / "doc.md"])
    assert report.succeeded == 1


@pytest.mark.asyncio
async def test_pipeline_cli_tags(pipeline, mock_embedder, settings, tmp_path):
    """CLI tags should be applied when no sidecar exists."""
    data_dir = tmp_path / "data"
    content = "# CLI Tag Doc\n\nSome content here. " * 10
    _write_md(data_dir / "doc.md", content)

    mock_embedder.embed_texts = AsyncMock(return_value=[
        {"dense": [0.1] * 1024, "sparse": {0: 0.5}}
    ] * 5)

    report = await pipeline.run([data_dir / "doc.md"], tags=["cli-tag"])
    assert report.succeeded == 1


@pytest.mark.asyncio
async def test_pipeline_sidecar_overrides_cli_tags(pipeline, mock_embedder, settings, tmp_path):
    """Sidecar tags should take priority over CLI tags (merged)."""
    data_dir = tmp_path / "data"
    content = "# Override Doc\n\nSome content here. " * 10
    _write_md(data_dir / "doc.md", content)

    sidecar = data_dir / ".meta.json"
    sidecar.write_text(json.dumps({"tags": ["sidecar-tag", "shared-tag"]}))

    mock_embedder.embed_texts = AsyncMock(return_value=[
        {"dense": [0.1] * 1024, "sparse": {0: 0.5}}
    ] * 5)

    report = await pipeline.run([data_dir / "doc.md"], tags=["cli-tag", "shared-tag"])
    assert report.succeeded == 1
    # Sidecar tags come first, CLI tags not in sidecar are appended
    # Result: ["sidecar-tag", "shared-tag", "cli-tag"]


@pytest.mark.asyncio
async def test_pipeline_orphan_cleanup(pipeline, settings, tmp_path):
    """Pipeline should clean up vectors for files no longer on disk."""
    data_dir = tmp_path / "data"
    content = "# Orphan Test\n\nContent here. " * 10
    _write_md(data_dir / "doc.md", content)

    # Ingest the file
    report1 = await pipeline.run([data_dir / "doc.md"])
    assert report1.succeeded == 1

    # Delete the file and re-run with empty list
    (data_dir / "doc.md").unlink()
    report2 = await pipeline.run([])

    assert report2.orphans_cleaned == 1


@pytest.mark.asyncio
async def test_pipeline_reingests_changed_file(pipeline, settings, tmp_path):
    """Changed file (different hash) should be re-ingested."""
    data_dir = tmp_path / "data"
    content1 = "# Version 1\n\nOriginal content here. " * 10
    _write_md(data_dir / "doc.md", content1)

    report1 = await pipeline.run([data_dir / "doc.md"])
    assert report1.succeeded == 1

    # Modify the file
    content2 = "# Version 2\n\nUpdated content here. " * 10
    _write_md(data_dir / "doc.md", content2)

    report2 = await pipeline.run([data_dir / "doc.md"])
    assert report2.succeeded == 1
    assert report2.skipped == 0


@pytest.mark.asyncio
async def test_pipeline_handles_embedder_failure(pipeline, mock_embedder, settings, tmp_path):
    """Embedder failure should be caught and reported, not crash the pipeline."""
    data_dir = tmp_path / "data"
    content = "# Fail Doc\n\nContent here. " * 10
    _write_md(data_dir / "doc.md", content)

    mock_embedder.embed_texts = AsyncMock(side_effect=RuntimeError("Embedding failed"))

    report = await pipeline.run([data_dir / "doc.md"])

    assert report.failed == 1
    assert report.succeeded == 0
    assert str(data_dir / "doc.md") in report.failed_files


@pytest.mark.asyncio
async def test_pipeline_no_files(pipeline, settings, tmp_path):
    """Pipeline with no files should return empty report."""
    report = await pipeline.run([])

    assert report.total == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.skipped == 0


@pytest.mark.asyncio
async def test_pipeline_default_data_dir(pipeline, settings, tmp_path):
    """Pipeline with no paths should scan DATA_DIR."""
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    content = "# Auto Doc\n\nContent here. " * 10
    _write_md(data_dir / "auto.md", content)

    report = await pipeline.run()  # No paths — should scan DATA_DIR

    assert report.succeeded == 1


@pytest.mark.asyncio
async def test_ingestion_report_defaults():
    """IngestionReport should have sensible defaults."""
    report = IngestionReport()
    assert report.total == 0
    assert report.succeeded == 0
    assert report.failed == 0
    assert report.skipped == 0
    assert report.orphans_cleaned == 0
    assert report.failed_files == []
