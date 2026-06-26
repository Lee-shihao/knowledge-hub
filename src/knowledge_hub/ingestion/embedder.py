import asyncio
import hashlib
import json
import os
from pathlib import Path

import structlog
import torch
from FlagEmbedding import BGEM3FlagModel

from knowledge_hub.config import Settings

logger = structlog.get_logger()


class OOMError(Exception):
    """Raised when the GPU runs out of memory during embedding."""
    pass


class FlagEmbeddingEmbedder:
    """Wraps FlagEmbedding's BGEM3FlagModel for dense + sparse embedding generation.

    Handles batch processing with OOM-aware auto-degradation:
    - Starts at configured batch_size (default 16)
    - On OOM: halves batch_size (min 4), persists to disk
    - After 3 batch failures: falls back to single-text serial embedding
    - Recovery: manual `kh config reset-batch-size`
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model_name = settings.EMBED_MODEL
        self._effective_batch = self._load_persisted_batch_size()

        # Resolve device
        device = self._resolve_device(settings.EMBED_DEVICE)
        use_fp16 = device == "cuda"

        logger.info(
            "Loading FlagEmbedding model",
            model=self._model_name,
            device=device,
            use_fp16=use_fp16,
        )

        # Pre-download model to local cache, excluding files that cause 403
        # errors on hf-mirror.com (.DS_Store, imgs/, onnx/).
        # Returns a local snapshot path so BGEM3FlagModel's internal
        # snapshot_download hits the cache instead of re-fetching from remote.
        local_path = self._ensure_model_cached(self._model_name)

        self._model = BGEM3FlagModel(
            local_path,
            use_fp16=use_fp16,
            device=device,
        )

    @staticmethod
    def _ensure_model_cached(model_name: str) -> str:
        """Pre-download model to local cache, excluding problematic files.

        Returns the local snapshot path so that BGEM3FlagModel's internal
        snapshot_download hits the cache and does not re-fetch files (like
        imgs/.DS_Store) that cause 403 errors on hf-mirror.com.

        The ignore list intentionally excludes:
        - .DS_Store at any depth (macOS junk in repo)
        - imgs/ directory (images accidentally committed, not needed for inference)
        - onnx/ directory (ONNX format, not needed for PyTorch inference)
        - Image files and non-PyTorch weight formats
        """
        from huggingface_hub import snapshot_download, constants

        hub_cache = Path(constants.HF_HUB_CACHE)

        try:
            # snapshot_download handles incremental caching internally —
            # if all non-ignored files are already cached, it returns quickly.
            snapshot_download(
                repo_id=model_name,
                endpoint=os.environ.get("HF_ENDPOINT"),  # explicit: env var not picked up by global constants
                ignore_patterns=[
                    "*.DS_Store",
                    "*/*.DS_Store",
                    "*/*/*.DS_Store",  # macOS system files at any depth
                    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp",
                    "*/*.jpg", "*/*.jpeg", "*/*.png", "*/*.gif", "*/*.webp",
                    "flax_model.msgpack", "rust_model.ot", "tf_model.h5",
                    "imgs/*",  # image directory accidentally in repo — not needed
                    "onnx/*",  # ONNX format — not needed for PyTorch inference
                ],
            )
        except Exception:
            logger.warning("Model pre-download failed, will retry on next load", model=model_name, exc_info=True)
            return model_name  # fallback: let BGEM3FlagModel try with the raw name

        # Resolve the local snapshot path so BGEM3FlagModel reuses the cache
        model_cache_name = f"models--{model_name.replace('/', '--')}"
        model_cache_path = hub_cache / model_cache_name
        snapshots_dir = model_cache_path / "snapshots"
        if snapshots_dir.exists():
            for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
                if snapshot.is_dir() and (snapshot / "config.json").exists():
                    logger.debug("Model cached at local path", model=model_name, path=str(snapshot))
                    return str(snapshot)

        # Fallback: return model name if path resolution fails
        logger.warning("Could not resolve local snapshot path, falling back to model name", model=model_name)
        return model_name

    @staticmethod
    def _resolve_device(embed_device: str) -> str:
        """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
        if embed_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return embed_device

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
                return await self._encode_batch(texts)
            except (OOMError, torch.cuda.OutOfMemoryError):
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
            results.extend(await self._encode_batch([t]))
        return results

    async def _encode_batch(self, texts: list[str]) -> list[dict]:
        """Encode texts using BGEM3FlagModel, returning dense + sparse vectors."""
        try:
            output = await asyncio.to_thread(
                self._model.encode,
                texts,
                return_dense=True,
                return_sparse=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            raise OOMError(str(e)) from e

        dense_vecs = output["dense_vecs"]  # shape: (N, 1024)
        lexical_weights = output["lexical_weights"]  # list[dict[str, float]]

        results = []
        for i in range(len(texts)):
            dense = dense_vecs[i].tolist()
            sparse = self._convert_sparse(lexical_weights[i])
            results.append({"dense": dense, "sparse": sparse})
        return results

    @staticmethod
    def _convert_sparse(lexical_weights: dict[str, float]) -> dict[int, float]:
        """Convert FlagEmbedding lexical_weights {str: float} to Qdrant-compatible {int: float}.

        Uses MD5 hash of the token string to produce a stable integer ID,
        consistent with the previous bag-of-words approach.
        """
        sparse: dict[int, float] = {}
        for token, weight in lexical_weights.items():
            token_id = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % 100000
            sparse[token_id] = weight
        return sparse

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
