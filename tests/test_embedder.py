import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import torch

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder, OOMError


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        EMBED_MODEL="BAAI/bge-m3",
        STORAGE_DIR=str(temp_storage_dir),
        EMBED_BATCH_SIZE=16,
        EMBED_DEVICE="cpu",  # Use CPU for tests
    )


@pytest.fixture
def embedder(settings):
    """Create embedder with mocked BGEM3FlagModel to avoid model download."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        embedder_instance = FlagEmbeddingEmbedder(settings)
        embedder_instance._model = mock_instance
        return embedder_instance


# ---- Integration tests (require FlagEmbedding model download) ----

@pytest.mark.asyncio
@pytest.mark.integration
async def test_embed_query_returns_dense_and_sparse(embedder):
    result = await embedder.embed_query("test query")
    assert "dense" in result
    assert "sparse" in result
    assert len(result["dense"]) == 1024
    assert isinstance(result["sparse"], dict)
    assert len(result["sparse"]) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_embed_texts_batch(embedder):
    texts = ["First text", "Second text", "Third text"]
    results = await embedder.embed_texts(texts)
    assert len(results) == 3
    for r in results:
        assert len(r["dense"]) == 1024
        assert isinstance(r["sparse"], dict)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_batch_size_persistence(embedder, temp_storage_dir):
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    embedder._effective_batch = 4
    embedder._persist_batch_size()
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["batch_size"] == 4


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reset_batch_size(embedder, temp_storage_dir):
    embedder._effective_batch = 4
    embedder._persist_batch_size()
    await embedder.reset_batch_size()
    assert embedder._effective_batch == 16
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    assert not state_file.exists()


# ---- Unit tests (mock-based, no model loading) ----

@pytest.mark.asyncio
async def test_embed_query_mocked(settings):
    """embed_query delegates to _embed_batch_with_retry and returns first result."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel"):
        embedder = FlagEmbeddingEmbedder(settings)
        fake_dense = [0.1] * 1024

        with patch.object(embedder, "_embed_batch_with_retry", new=AsyncMock()) as mock_batch:
            mock_batch.return_value = [
                {"dense": fake_dense, "sparse": {1234: 1.0}}
            ]
            result = await embedder.embed_query("hello")
            mock_batch.assert_awaited_once_with(["hello"])
            assert result["dense"] == fake_dense
            assert result["sparse"] == {1234: 1.0}


@pytest.mark.asyncio
async def test_embed_texts_splits_into_batches(settings):
    """embed_texts splits texts into chunks of _effective_batch."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel"):
        embedder = FlagEmbeddingEmbedder(settings)
        embedder._effective_batch = 2

        call_count = 0
        async def fake_batch(texts):
            nonlocal call_count
            call_count += 1
            return [{"dense": [0.1] * 1024, "sparse": {i: 1.0}} for i in range(len(texts))]

        with patch.object(embedder, "_embed_batch_with_retry", new=AsyncMock()) as mock_batch:
            mock_batch.side_effect = fake_batch
            results = await embedder.embed_texts(["a", "b", "c", "d", "e"])
            assert call_count == 3  # 2+2+1
            assert len(results) == 5


@pytest.mark.asyncio
async def test_encode_batch_success(settings):
    """_encode_batch returns dense + sparse for each input text."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance

        # Mock the encode output
        import numpy as np
        fake_dense = np.array([[0.5] * 1024, [0.6] * 1024])
        fake_lexical = [{"token1": 0.8, "token2": 0.3}, {"token3": 0.9}]
        mock_instance.encode.return_value = {
            "dense_vecs": fake_dense,
            "lexical_weights": fake_lexical,
        }

        embedder = FlagEmbeddingEmbedder(settings)
        results = await embedder._encode_batch(["text a", "text b"])

        assert len(results) == 2
        assert len(results[0]["dense"]) == 1024
        assert results[0]["dense"][0] == 0.5
        assert results[1]["dense"][0] == 0.6
        assert isinstance(results[0]["sparse"], dict)
        assert len(results[0]["sparse"]) == 2  # token1 and token2


@pytest.mark.asyncio
async def test_encode_batch_oom_detection(settings):
    """_encode_batch should raise OOMError when GPU runs out of memory."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance

        # Mock encode to raise OOM
        mock_instance.encode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

        embedder = FlagEmbeddingEmbedder(settings)
        with pytest.raises(OOMError, match="CUDA out of memory"):
            await embedder._encode_batch(["text a"])


@pytest.mark.asyncio
async def test_embed_batch_with_retry_oom_then_serial_fallback(settings):
    """On repeated OOM, falls back to serial single-text calls."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel"):
        embedder = FlagEmbeddingEmbedder(settings)

        call_count = 0

        async def encode_side_effect(texts):
            nonlocal call_count
            call_count += 1
            raise OOMError("mock oom")

        with patch.object(embedder, "_encode_batch", new=AsyncMock()) as mock_encode:
            mock_encode.side_effect = encode_side_effect
            # 3 texts -> 3 batch attempts (all OOM), then serial: 3 single calls = 6 total ooms
            with pytest.raises(OOMError):
                await embedder._embed_batch_with_retry(["a", "b", "c"])
            # All calls failed with OOM
            assert call_count >= 3


@pytest.mark.asyncio
async def test_embed_batch_success_after_one_oom(settings):
    """After one OOM, batch size is halved and retry succeeds."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel"):
        embedder = FlagEmbeddingEmbedder(settings)
        embedder._effective_batch = 8

        call_count = 0

        async def encode_side_effect(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OOMError("mock oom")
            # On retry, return success
            return [{"dense": [0.1] * 1024, "sparse": {1: 1.0}} for _ in texts]

        with patch.object(embedder, "_encode_batch", new=AsyncMock()) as mock_encode:
            mock_encode.side_effect = encode_side_effect

            with patch.object(embedder, "_persist_batch_size"):
                results = await embedder._embed_batch_with_retry(["a", "b"])
                assert len(results) == 2
                assert embedder._effective_batch == 4  # halved from 8
                assert call_count == 2


def test_convert_sparse():
    """_convert_sparse converts lexical_weights {str: float} to {int: float} via MD5."""
    lexical = {"hello": 0.8, "world": 0.5}
    result = FlagEmbeddingEmbedder._convert_sparse(lexical)

    assert isinstance(result, dict)
    assert len(result) == 2

    # Check that keys are integers
    for key in result.keys():
        assert isinstance(key, int)

    # Check values are preserved
    assert set(result.values()) == {0.8, 0.5}


def test_convert_sparse_empty():
    """_convert_sparse on empty dict returns empty dict."""
    result = FlagEmbeddingEmbedder._convert_sparse({})
    assert result == {}


def test_persist_batch_size_writes_file(embedder, temp_storage_dir):
    """_persist_batch_size writes JSON to .batch_size_state.json."""
    embedder._effective_batch = 8
    embedder._persist_batch_size()
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["batch_size"] == 8


def test_load_persisted_batch_size_reads_file(embedder, temp_storage_dir):
    """_load_persisted_batch_size returns the persisted value."""
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    state_file.write_text(json.dumps({"batch_size": 4}))
    result = embedder._load_persisted_batch_size()
    assert result == 4


def test_load_persisted_batch_size_default(settings):
    """When no state file exists, falls back to EMBED_BATCH_SIZE."""
    with patch("knowledge_hub.ingestion.embedder.BGEM3FlagModel"):
        embedder = FlagEmbeddingEmbedder(settings)
        assert embedder._effective_batch == settings.EMBED_BATCH_SIZE


@pytest.mark.asyncio
async def test_reset_batch_size_removes_file(embedder, temp_storage_dir):
    """reset_batch_size restores EMBED_BATCH_SIZE and deletes state file."""
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    state_file.write_text(json.dumps({"batch_size": 4}))
    embedder._effective_batch = 4
    await embedder.reset_batch_size()
    assert embedder._effective_batch == 16
    assert not state_file.exists()


def test_resolve_device_auto_cuda(settings):
    """_resolve_device returns 'cuda' when available."""
    with patch("knowledge_hub.ingestion.embedder.torch.cuda.is_available", return_value=True):
        assert FlagEmbeddingEmbedder._resolve_device("auto") == "cuda"


def test_resolve_device_auto_cpu(settings):
    """_resolve_device returns 'cpu' when CUDA is not available."""
    with patch("knowledge_hub.ingestion.embedder.torch.cuda.is_available", return_value=False):
        assert FlagEmbeddingEmbedder._resolve_device("auto") == "cpu"


def test_resolve_device_explicit(settings):
    """_resolve_device returns explicit device when not 'auto'."""
    assert FlagEmbeddingEmbedder._resolve_device("cpu") == "cpu"
    assert FlagEmbeddingEmbedder._resolve_device("cuda") == "cuda"
