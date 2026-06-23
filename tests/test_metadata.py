# tests/test_metadata.py
import pytest
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.storage.metadata import SourceMetadataManager


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_knowledge_hub",
        STORAGE_DIR=str(temp_storage_dir),
    )


@pytest.fixture
async def metadata_mgr(settings):
    client = QdrantClient(settings.QDRANT_URL)
    mgr = SourceMetadataManager(settings, client)
    await mgr.ensure_collection()
    yield mgr
    # Cleanup
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upsert_and_get_hash(metadata_mgr):
    await metadata_mgr.upsert("test.pdf", "abc123", 10)
    h = await metadata_mgr.get_hash("test.pdf")
    assert h == "abc123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_hash_missing(metadata_mgr):
    h = await metadata_mgr.get_hash("nonexistent.pdf")
    assert h is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_sources(metadata_mgr):
    await metadata_mgr.upsert("a.pdf", "h1", 5)
    await metadata_mgr.upsert("b.pdf", "h2", 3)
    sources = await metadata_mgr.list_sources()
    assert set(sources) == {"a.pdf", "b.pdf"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove(metadata_mgr):
    await metadata_mgr.upsert("x.pdf", "h1", 1)
    await metadata_mgr.remove("x.pdf")
    assert await metadata_mgr.get_hash("x.pdf") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_cleanup(metadata_mgr):
    await metadata_mgr.upsert("keep.pdf", "h1", 5)
    await metadata_mgr.upsert("orphan.pdf", "h2", 3)
    removed = await metadata_mgr.orphan_cleanup({"keep.pdf", "other.pdf"})
    assert removed == 1  # orphan.pdf removed
    assert await metadata_mgr.get_hash("orphan.pdf") is None
    assert await metadata_mgr.get_hash("keep.pdf") == "h1"
