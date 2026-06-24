import hashlib
import pytest
from qdrant_client import QdrantClient

from knowledge_hub.config import Settings
from knowledge_hub.schemas import DocumentChunk, ChunkMetadata
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager


def make_chunk_id(source_file: str, heading_path: list[str], text: str) -> str:
    joined = "|".join(heading_path)
    raw = f"{source_file}_{joined}_{text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_kb_vec",
        STORAGE_DIR=str(temp_storage_dir),
    )


@pytest.fixture
async def vector_store(settings):
    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    yield store
    # Cleanup
    client.delete_collection(settings.QDRANT_COLLECTION)
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.asyncio
async def test_ensure_collection(vector_store):
    """Collection should exist without error after ensure_collection."""
    # Second call should be idempotent
    await vector_store.ensure_collection()


@pytest.mark.asyncio
async def test_upsert_and_count(vector_store):
    chunk = DocumentChunk(
        id=make_chunk_id("test.pdf", ["Ch1"], "Some text content for testing purposes"),
        text="Some text content for testing purposes",
        dense_embedding=[0.1] * 1024,
        sparse_embedding={0: 0.5, 10: 0.3, 100: 0.8},
        metadata=ChunkMetadata(
            source_file="test.pdf",
            source_hash="abc123",
            heading_path=["Ch1"],
            tags=["test"],
        ),
    )
    await vector_store.upsert_chunks([chunk])
    c = await vector_store.count()
    assert c == 1


@pytest.mark.asyncio
async def test_upsert_idempotent(vector_store):
    """Upserting with same ID should overwrite, not duplicate."""
    chunk = DocumentChunk(
        id=make_chunk_id("test.pdf", ["Ch1"], "content"),
        text="content",
        dense_embedding=[0.0] * 1024,
        sparse_embedding={0: 1.0},
        metadata=ChunkMetadata(source_file="test.pdf", source_hash="h1"),
    )
    await vector_store.upsert_chunks([chunk])
    await vector_store.upsert_chunks([chunk])
    assert await vector_store.count() == 1


@pytest.mark.asyncio
async def test_delete_by_source(vector_store):
    chunk = DocumentChunk(
        id=make_chunk_id("del.pdf", ["Ch1"], "to delete"),
        text="to delete",
        dense_embedding=[0.0] * 1024,
        sparse_embedding={0: 1.0},
        metadata=ChunkMetadata(source_file="del.pdf", source_hash="h1"),
    )
    await vector_store.upsert_chunks([chunk])
    await vector_store.delete_by_source("del.pdf")
    assert await vector_store.count() == 0


@pytest.mark.asyncio
async def test_hybrid_search_no_filter(vector_store):
    """Hybrid search should return results with scores."""
    chunk = DocumentChunk(
        id=make_chunk_id("search.pdf", ["Intro"], "Searchable content here"),
        text="Searchable content here",
        dense_embedding=[0.5] * 1024,
        sparse_embedding={5: 0.9, 20: 0.7},
        metadata=ChunkMetadata(
            source_file="search.pdf",
            source_hash="hash1",
            heading_path=["Intro"],
            tags=["demo"],
        ),
    )
    await vector_store.upsert_chunks([chunk])

    results = await vector_store.hybrid_search(
        dense_vec=[0.5] * 1024,
        sparse_vec={5: 0.9, 20: 0.7},
        top_k=10,
    )
    assert len(results) == 1
    chunk_id, score, payload = results[0]
    # Qdrant returns UUID with hyphens; normalize for comparison
    assert chunk_id.replace("-", "") == chunk.id.replace("-", "")
    assert score > 0
    assert payload["text"] == "Searchable content here"
    assert payload["source_file"] == "search.pdf"


@pytest.mark.asyncio
async def test_hybrid_search_with_source_filter(vector_store):
    """Hybrid search should filter by source file."""
    chunk1 = DocumentChunk(
        id=make_chunk_id("doc1.pdf", ["A"], "Content from doc1"),
        text="Content from doc1",
        dense_embedding=[0.2] * 1024,
        sparse_embedding={1: 0.5},
        metadata=ChunkMetadata(source_file="doc1.pdf", source_hash="h1"),
    )
    chunk2 = DocumentChunk(
        id=make_chunk_id("doc2.pdf", ["B"], "Content from doc2"),
        text="Content from doc2",
        dense_embedding=[0.2] * 1024,
        sparse_embedding={1: 0.5},
        metadata=ChunkMetadata(source_file="doc2.pdf", source_hash="h2"),
    )
    await vector_store.upsert_chunks([chunk1, chunk2])

    results = await vector_store.hybrid_search(
        dense_vec=[0.2] * 1024,
        sparse_vec={1: 0.5},
        top_k=10,
        filter_source="doc1.pdf",
    )
    assert len(results) == 1
    _, _, payload = results[0]
    assert payload["source_file"] == "doc1.pdf"


@pytest.mark.asyncio
async def test_hybrid_search_with_tag_filter(vector_store):
    """Hybrid search should filter by tags."""
    chunk1 = DocumentChunk(
        id=make_chunk_id("tagged.pdf", ["A"], "Tagged content"),
        text="Tagged content",
        dense_embedding=[0.3] * 1024,
        sparse_embedding={2: 0.6},
        metadata=ChunkMetadata(
            source_file="tagged.pdf",
            source_hash="h1",
            tags=["important", "reviewed"],
        ),
    )
    chunk2 = DocumentChunk(
        id=make_chunk_id("untagged.pdf", ["B"], "Untagged content"),
        text="Untagged content",
        dense_embedding=[0.3] * 1024,
        sparse_embedding={2: 0.6},
        metadata=ChunkMetadata(
            source_file="untagged.pdf",
            source_hash="h2",
            tags=["draft"],
        ),
    )
    await vector_store.upsert_chunks([chunk1, chunk2])

    results = await vector_store.hybrid_search(
        dense_vec=[0.3] * 1024,
        sparse_vec={2: 0.6},
        top_k=10,
        filter_tags=["important"],
    )
    assert len(results) == 1
    _, _, payload = results[0]
    assert "important" in payload["tags"]


@pytest.mark.asyncio
async def test_upsert_multiple_chunks(vector_store):
    """Upsert should handle multiple chunks in a batch."""
    chunks = [
        DocumentChunk(
            id=make_chunk_id("multi.pdf", [f"Ch{i}"], f"Content {i}"),
            text=f"Content {i}",
            dense_embedding=[float(i) / 10] * 1024,
            sparse_embedding={i: float(i) / 10},
            metadata=ChunkMetadata(
                source_file="multi.pdf",
                source_hash=f"hash{i}",
            ),
        )
        for i in range(5)
    ]
    await vector_store.upsert_chunks(chunks)
    assert await vector_store.count() == 5
