import asyncio
import hashlib
import json
import re
from pathlib import Path

import httpx
import structlog

from knowledge_hub.config import Settings

logger = structlog.get_logger()


class OOMError(Exception):
    """Raised when Ollama returns an out-of-memory error."""
    pass


class OllamaEmbedder:
    """Wraps Ollama's bge-m3 for dense + sparse embedding generation.

    Handles batch processing with OOM-aware auto-degradation:
    - Starts at configured batch_size (default 16)
    - On OOM: halves batch_size (min 4), persists to disk
    - After 3 batch failures: falls back to single-text serial embedding
    - Recovery: manual `kh config reset-batch-size`
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.EMBED_MODEL
        self._effective_batch = self._load_persisted_batch_size()

    async def embed_query(self, query: str) -> dict:
        """Embed a single query text. Returns {dense, sparse}."""
        results = await self._embed_batch_with_retry([query])
        return results[0]

    async def embed_texts(self, texts: list[str]) -> list[dict]:
        """Embed a batch of texts. Each result is {dense, sparse}."""
        all_results = []
        for i in range(0, len(texts), self._effective_batch):
            batch = texts[i:i + self._effective_batch]
            batch_results = await self._embed_batch_with_retry(batch)
            all_results.extend(batch_results)
        return all_results

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[dict]:
        for attempt in range(3):
            try:
                return await self._call_ollama(texts)
            except OOMError:
                self._effective_batch = max(4, self._effective_batch // 2)
                self._persist_batch_size()
                logger.warning(
                    "OOM detected, degraded batch_size",
                    new_batch_size=self._effective_batch,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(2 ** attempt)
                continue

        # All batch attempts failed — serial fallback
        logger.warning("Falling back to single-text serial embedding")
        results = []
        for t in texts:
            results.extend(await self._call_ollama([t]))
        return results

    async def _call_ollama(self, texts: list[str]) -> list[dict]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            if response.status_code != 200:
                if "out of memory" in response.text.lower():
                    raise OOMError(response.text)
                raise RuntimeError(f"Ollama API error: {response.status_code} {response.text}")
            data = response.json()
            # bge-m3 via Ollama returns "embeddings" as list[list[float]]
            # The sparse representation is extracted from the model's output
            # Ollama's /api/embed doesn't natively return sparse; we extract it
            # by requesting bge-m3's sparse embedding via a separate mechanism.
            # For now, use dense-only and generate a simple sparse bag-of-words.
            return [
                {
                    "dense": emb,
                    "sparse": self._sparse_bow(texts[i]),
                }
                for i, emb in enumerate(data["embeddings"])
            ]

    def _sparse_bow(self, text: str) -> dict[int, float]:
        """Generate a simple bag-of-words sparse vector as fallback.

        NOTE: Ollama's /api/embed endpoint currently only returns dense
        embeddings. bge-m3's native sparse vectors (lexical weights per token)
        are not exposed. This BoW fallback provides a functional keyword-match
        signal for hybrid search in the MVP.

        Future upgrade path: use a separate sparse encoder (e.g., direct
        llama-cpp-python with bge-m3, or a dedicated sparse model) and
        swap this method.
        """
        tokens = re.findall(r'\w+', text.lower())
        sparse = {}
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % 100000
            sparse[h] = sparse.get(h, 0.0) + 1.0
        # Normalize
        max_val = max(sparse.values()) if sparse else 1.0
        return {k: v / max_val for k, v in sparse.items()}

    def _persist_batch_size(self):
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"batch_size": self._effective_batch}))

    def _load_persisted_batch_size(self) -> int:
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())["batch_size"]
        return self.settings.EMBED_BATCH_SIZE

    async def reset_batch_size(self) -> None:
        self._effective_batch = self.settings.EMBED_BATCH_SIZE
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            state_file.unlink()
