"""Reranker using FlagEmbedding's FlagReranker for cross-encoder re-ranking."""
import asyncio
import structlog
import torch
from FlagEmbedding import FlagReranker

from knowledge_hub.config import Settings

logger = structlog.get_logger()


class Reranker:
    """Cross-encoder reranker using bge-reranker-v2-m3 via FlagEmbedding.

    Re-ranks candidate documents using a cross-encoder model for improved relevance.
    On any failure (OOM, model error, etc.), gracefully degrades by returning
    the original candidates unchanged.

    Attributes:
        _model: The FlagReranker model instance.
    """

    def __init__(self, settings: Settings):
        """Initialize the reranker with the configured model.

        Args:
            settings: Application settings containing RERANK_MODEL and EMBED_DEVICE.
        """
        self._model_name = settings.RERANK_MODEL
        device = self._resolve_device(settings.EMBED_DEVICE)
        use_fp16 = device == "cuda"

        logger.info(
            "Loading FlagReranker model",
            model=self._model_name,
            device=device,
            use_fp16=use_fp16,
        )
        self._model = FlagReranker(
            self._model_name,
            use_fp16=use_fp16,
            device=device,
        )

    @staticmethod
    def _resolve_device(embed_device: str) -> str:
        """Resolve 'auto' to 'cuda' if available, else 'cpu'.

        Args:
            embed_device: Device string from settings ('auto', 'cpu', or 'cuda').

        Returns:
            Resolved device string.
        """
        if embed_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return embed_device

    async def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5
    ) -> list[dict]:
        """Re-rank candidates by relevance to the query.

        Uses FlagReranker's compute_score to get relevance scores, then returns
        the top_k candidates sorted by score in descending order.

        On any exception (OOM, model error, etc.), returns the original candidates
        capped to top_k (graceful degradation).

        Args:
            query: The search query string.
            candidates: List of candidate dicts with at minimum 'text' and 'score' keys.
                       Other metadata keys (source_file, heading_path, etc.) are preserved.
            top_k: Maximum number of candidates to return.

        Returns:
            Top_k candidates sorted by reranker score, or original candidates on failure.
        """
        if not candidates:
            return []

        try:
            # Prepare query-document pairs for scoring
            pairs = [[query, c["text"]] for c in candidates]

            # Compute scores (synchronous call wrapped in to_thread)
            scores = await asyncio.to_thread(
                self._model.compute_score,
                pairs,
            )

            # Handle both single score and list of scores
            if isinstance(scores, (int, float)):
                scores = [scores]

            # Attach scores to candidates
            scored_candidates = []
            for candidate, score in zip(candidates, scores):
                scored_candidates.append({**candidate, "score": float(score)})

            # Sort by score descending and return top_k
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            return scored_candidates[:top_k]

        except Exception as e:
            logger.warning(
                "reranker_failed_degrading",
                error=str(e),
                model=self._model_name,
            )
            # Graceful degradation: return candidates unchanged, capped to top_k
            return candidates[:top_k]