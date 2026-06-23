import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import httpx

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import OllamaEmbedder, OOMError


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        OLLAMA_BASE_URL="http://localhost:11434",
        EMBED_MODEL="bge-m3",
        STORAGE_DIR=str(temp_storage_dir),
        EMBED_BATCH_SIZE=16,
    )


@pytest.fixture
def embedder(settings):
    return OllamaEmbedder(settings)


# ---- Integration tests (require Ollama) ----

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


# ---- Unit tests (mock-based, no Ollama needed) ----

@pytest.mark.asyncio
async def test_embed_query_mocked(settings):
    """embed_query delegates to _embed_batch_with_retry and returns first result."""
    embedder = OllamaEmbedder(settings)
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
    embedder = OllamaEmbedder(settings)
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
async def test_call_ollama_success(settings):
    """_call_ollama returns dense + sparse for each input text."""
    embedder = OllamaEmbedder(settings)
    fake_dense = [[0.5] * 1024, [0.6] * 1024]

    mock_response = httpx.Response(
        status_code=200,
        json={"embeddings": fake_dense},
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("knowledge_hub.ingestion.embedder.httpx.AsyncClient", return_value=mock_client):
        results = await embedder._call_ollama(["text a", "text b"])

    assert len(results) == 2
    assert results[0]["dense"] == [0.5] * 1024
    assert results[1]["dense"] == [0.6] * 1024
    assert isinstance(results[0]["sparse"], dict)
    assert len(results[0]["sparse"]) > 0  # "text" and "a" tokens


@pytest.mark.asyncio
async def test_call_ollama_oom_detection(settings):
    """_call_ollama should raise OOMError when Ollama reports out of memory."""
    embedder = OllamaEmbedder(settings)

    mock_response = httpx.Response(
        status_code=500,
        text="CUDA out of memory while allocating embeddings",
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("knowledge_hub.ingestion.embedder.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(OOMError):
            await embedder._call_ollama(["text a"])


@pytest.mark.asyncio
async def test_call_ollama_other_error(settings):
    """_call_ollama raises RuntimeError for non-OOM API errors."""
    embedder = OllamaEmbedder(settings)

    mock_response = httpx.Response(status_code=404, text="model not found")
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("knowledge_hub.ingestion.embedder.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Ollama API error"):
            await embedder._call_ollama(["text a"])


@pytest.mark.asyncio
async def test_embed_batch_with_retry_oom_then_serial_fallback(settings):
    """On repeated OOM, falls back to serial single-text calls."""
    embedder = OllamaEmbedder(settings)

    # First 3 calls (batch attempts) raise OOM
    # Downside: the method halves _effective_batch (min 4) then calls _call_ollama again
    # So after the 3 batch failures, it enters the serial fallback loop.
    call_count = 0

    async def call_side_effect(texts):
        nonlocal call_count
        call_count += 1
        raise OOMError("mock oom")

    with patch.object(embedder, "_call_ollama", new=AsyncMock()) as mock_call:
        mock_call.side_effect = call_side_effect
        # 3 texts -> 3 batch attempts (all OOM), then serial: 3 single calls = 6 total ooms
        # The method raises OOMError each time and eventually the serial loop also fails
        with pytest.raises(OOMError):
            await embedder._embed_batch_with_retry(["a", "b", "c"])
        # All calls failed with OOM
        assert call_count >= 3


@pytest.mark.asyncio
async def test_embed_batch_success_after_one_oom(settings):
    """After one OOM, batch size is halved and retry succeeds."""
    embedder = OllamaEmbedder(settings)
    embedder._effective_batch = 8

    call_count = 0

    async def call_side_effect(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OOMError("mock oom")
        # On retry, return success
        return [{"dense": [0.1] * 1024, "sparse": {1: 1.0}} for _ in texts]

    with patch.object(embedder, "_call_ollama", new=AsyncMock()) as mock_call:
        mock_call.side_effect = call_side_effect

        # Also need to patch _persist_batch_size to avoid file I/O
        with patch.object(embedder, "_persist_batch_size"):
            results = await embedder._embed_batch_with_retry(["a", "b"])
            assert len(results) == 2
            assert embedder._effective_batch == 4  # halved from 8
            assert call_count == 2


def test_sparse_bow_generates_tokens(embedder):
    """_sparse_bow returns a normalized token frequency dict."""
    result = embedder._sparse_bow("hello world hello")
    assert isinstance(result, dict)
    # "hello" appears twice, "world" once; max = 2, so hello -> 1.0, world -> 0.5
    hello_hash = hash("hello") % 100000
    world_hash = hash("world") % 100000
    assert result[hello_hash] == 1.0
    assert result[world_hash] == 0.5


def test_sparse_bow_empty_string(embedder):
    """_sparse_bow on empty string returns empty dict."""
    result = embedder._sparse_bow("")
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
    embedder = OllamaEmbedder(settings)
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
