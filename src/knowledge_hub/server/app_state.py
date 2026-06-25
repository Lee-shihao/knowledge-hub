"""AppState — single-process shared state for MCP and HTTP servers.

All GPU-heavy components (embedder, reranker) are loaded once and shared.
"""
from __future__ import annotations

from dataclasses import dataclass

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.pipeline import IngestionPipeline
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.storage.vector_store import (
    QdrantVectorStore,
    build_qdrant_client,
)
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.server.job_manager import JobManager


@dataclass
class AppState:
    """Shared application state for all servers in the process.

    Holds references to all initialized components so that MCP and HTTP
    servers share one copy of the embedding model, reranker, and Qdrant.
    """

    settings: Settings
    qdrant_client: "QdrantClient"
    embedder: FlagEmbeddingEmbedder
    reranker: Reranker
    metadata_mgr: SourceMetadataManager
    vector_store: QdrantVectorStore
    query_engine: QueryEngine
    pipeline: IngestionPipeline
    health: HealthMonitor
    job_manager: JobManager
    mcp: "FastMCP" | None = None  # Set during create()

    @classmethod
    async def create(cls, settings: Settings) -> "AppState":
        """Build all components and return a fully-initialized AppState.

        Two-phase construction: components first, then AppState instance
        with mcp=None, then mcp is constructed with a self-reference.

        Args:
            settings: Application settings.

        Returns:
            Fully initialized AppState with mcp created.
        """
        from qdrant_client import QdrantClient

        # Phase 1: Build all components
        client = build_qdrant_client(settings)
        embedder = FlagEmbeddingEmbedder(settings)
        reranker = Reranker(settings)
        metadata_mgr = SourceMetadataManager(settings, client)
        vector_store = QdrantVectorStore(settings, client, metadata_mgr)
        query_engine = QueryEngine(
            settings, embedder, vector_store, reranker
        )
        pipeline = IngestionPipeline(
            settings,
            DocumentLoader(settings),
            SemanticChunker(settings),
            embedder,
            vector_store,
            metadata_mgr,
        )
        job_manager = JobManager(pipeline)
        health = HealthMonitor(settings, client)
        await health.start()

        # Phase 2: Build state, then create mcp with self-reference
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = cls(
            settings=settings,
            qdrant_client=client,
            embedder=embedder,
            reranker=reranker,
            metadata_mgr=metadata_mgr,
            vector_store=vector_store,
            query_engine=query_engine,
            pipeline=pipeline,
            health=health,
            job_manager=job_manager,
            mcp=None,
        )
        state.mcp = create_mcp_app(state)
        return state
