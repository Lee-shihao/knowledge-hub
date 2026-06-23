import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Temporary storage directory for Qdrant and batch_size state."""
    d = tmp_path / "storage"
    d.mkdir()
    return d
