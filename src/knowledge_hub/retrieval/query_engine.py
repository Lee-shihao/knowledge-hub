"""QueryEngine — orchestrates the full query flow: embed → hybrid search → rerank."""
import time

import structlog

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.schemas import QueryInput, QueryResult, ChunkResult
from knowledge_hub.storage.vector_store import QdrantVectorStore

logger = structlog.get_logger()


class QueryEngine:
    """Orchestrates the full query flow: embed → hybrid search → rerank.

    Depends on FlagEmbeddingEmbedder for query embedding, QdrantVectorStore
    for hybrid search, and Reranker for cross-encoder reranking.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: FlagEmbeddingEmbedder,
        vector_store: QdrantVectorStore,
        reranker: Reranker,
    ):
        self.settings = settings
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker

    async def query(self, q: QueryInput) -> QueryResult:
        """Execute a query: embed → hybrid search → rerank → build result.

        Args:
            q: Query input with query text, top_k, and optional filters.

        Returns:
            QueryResult with ranked ChunkResult items and timing.
        """
        start = time.perf_counter()

        # 1. Embed the query
        query_embedding = await self._embedder.embed_query(q.query)

        # 2. Hybrid search (get HYBRID_CANDIDATE_K candidates)
        candidates = await self._store.hybrid_search(
            dense_vec=query_embedding["dense"],
            sparse_vec=query_embedding["sparse"],
            top_k=self.settings.HYBRID_CANDIDATE_K,
            filter_source=q.filter_source,
            filter_tags=q.filter_tags,
        )

        if not candidates:
            elapsed = (time.perf_counter() - start) * 1000
            return QueryResult(results=[], query_time_ms=elapsed)

        # 3. Rerank — flatten payload into candidate dict
        candidate_dicts = [
            {"text": payload.get("text", ""), "score": score, **payload}
            for (_id, score, payload) in candidates
        ]
        reranked = await self._reranker.rerank(q.query, candidate_dicts, top_k=q.top_k)

        # 4. Build QueryResult with ChunkResult items
        results = [
            ChunkResult(
                text=c["text"],
                source_file=c.get("source_file", "unknown"),
                page_or_section=_build_page_or_section(c),
                heading_path=c.get("heading_path", []),
                score=c["score"],
            )
            for c in reranked
        ]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "query_complete",
            query=q.query[:80],
            results=len(results),
            time_ms=elapsed,
        )
        return QueryResult(results=results, query_time_ms=elapsed)


def _build_page_or_section(candidate: dict) -> str:
    """Build page_or_section from candidate dict.

    Uses heading_path[-1] if available, else f'p{page_number}' if page_number
    exists, else empty string.
    """
    heading_path = candidate.get("heading_path", [])
    if heading_path:
        return heading_path[-1]

    page_number = candidate.get("page_number")
    if page_number is not None:
        return f"p{page_number}"

    return ""
