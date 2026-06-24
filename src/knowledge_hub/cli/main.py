"""CLI for knowledge-hub: index, query, manage, and serve.

Heavy imports (FlagEmbedding, torch, llama_index) are deferred to command
execution time so that `kh --help` and `kh config show` respond instantly.
"""

import asyncio
from pathlib import Path

import click
import structlog

from knowledge_hub.config import Settings

logger = structlog.get_logger()


def _get_settings() -> Settings:
    return Settings()


def _build_pipeline(settings):
    """Build ingestion pipeline (triggers heavy imports)."""
    from qdrant_client import QdrantClient
    from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
    from knowledge_hub.ingestion.loaders import DocumentLoader
    from knowledge_hub.ingestion.chunker import SemanticChunker
    from knowledge_hub.ingestion.pipeline import IngestionPipeline
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import QdrantVectorStore

    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    meta_mgr = SourceMetadataManager(settings, client)
    store = QdrantVectorStore(settings, client, meta_mgr)
    return IngestionPipeline(
        settings=settings,
        loader=DocumentLoader(settings),
        chunker=SemanticChunker(settings),
        embedder=FlagEmbeddingEmbedder(settings),
        vector_store=store,
        metadata_mgr=meta_mgr,
    )


def _build_query_engine(settings):
    """Build query engine (triggers heavy imports)."""
    from qdrant_client import QdrantClient
    from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
    from knowledge_hub.retrieval.query_engine import QueryEngine
    from knowledge_hub.retrieval.reranker import Reranker
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import QdrantVectorStore

    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    meta_mgr = SourceMetadataManager(settings, client)
    store = QdrantVectorStore(settings, client, meta_mgr)
    embedder = FlagEmbeddingEmbedder(settings)
    reranker = Reranker(settings)
    return QueryEngine(settings, embedder, store, reranker)


@click.group()
def cli():
    """knowledge-hub — Local Vector RAG knowledge base."""
    pass


@cli.command()
@click.option("--path", type=click.Path(exists=True), default=None,
              help="Directory to ingest from.")
@click.option("--force", is_flag=True,
              help="Re-ingest all files, ignoring source hash cache.")
@click.option("--tags", default=None,
              help="Comma-separated tags (overridden by .meta.json sidecars).")
def index(path, force, tags):
    """Ingest documents into the knowledge base."""
    settings = _get_settings()
    pipeline = _build_pipeline(settings)
    paths = [Path(path)] if path else None
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    report = asyncio.run(pipeline.run(paths=paths, force=force, tags=tag_list))
    click.echo(
        f"Total: {report.total}, Succeeded: {report.succeeded}, "
        f"Failed: {report.failed}, Skipped: {report.skipped}, "
        f"Orphans cleaned: {report.orphans_cleaned}"
    )
    if report.failed_files:
        click.echo("Failed files:")
        for f in report.failed_files:
            click.echo(f"  - {f}")


@cli.command()
@click.argument("query_text")
@click.option("-k", "--top-k", type=int, default=5, help="Number of results.")
def query(query_text, top_k):
    """Query the knowledge base directly."""
    from knowledge_hub.schemas import QueryInput

    settings = _get_settings()
    engine = _build_query_engine(settings)
    result = asyncio.run(engine.query(QueryInput(query=query_text, top_k=top_k)))
    click.echo(f"Results ({result.query_time_ms:.1f}ms):")
    for i, r in enumerate(result.results):
        heading = " > ".join(r.heading_path) if r.heading_path else "(no heading)"
        click.echo(f"\n--- Result {i+1} (score: {r.score:.3f}) ---")
        click.echo(f"Source: {r.source_file} | {r.page_or_section} | {heading}")
        click.echo(r.text[:500])


@cli.command()
def status():
    """Show knowledge base status."""
    from qdrant_client import QdrantClient
    from knowledge_hub.storage.metadata import SourceMetadataManager

    settings = _get_settings()
    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    try:
        count = client.count(collection_name=settings.QDRANT_COLLECTION).count
        meta_mgr = SourceMetadataManager(settings, client)
        sources = asyncio.run(meta_mgr.list_sources())
        click.echo(f"Collection: {settings.QDRANT_COLLECTION}")
        click.echo(f"Total chunks: {count}")
        click.echo(f"Source files: {len(sources)}")
        if sources:
            click.echo("Sources:")
            for s in sorted(sources)[:20]:
                click.echo(f"  - {s}")
            if len(sources) > 20:
                click.echo(f"  ... and {len(sources) - 20} more")
    except Exception as e:
        click.echo(f"Error connecting to Qdrant: {e}")


@cli.command()
def cleanup_orphans():
    """Remove vectors for deleted source files."""
    from qdrant_client import QdrantClient
    from knowledge_hub.storage.metadata import SourceMetadataManager

    settings = _get_settings()
    data_dir = Path(settings.DATA_DIR)
    local_files = {p.name for p in data_dir.rglob("*") if p.is_file()} if data_dir.exists() else set()
    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    meta_mgr = SourceMetadataManager(settings, client)
    removed = asyncio.run(meta_mgr.orphan_cleanup(local_files))
    click.echo(f"Removed {removed} orphan source(s).")


@cli.group()
def config():
    """Manage configuration."""
    pass


@config.command("show")
def config_show():
    """Show current effective configuration."""
    settings = _get_settings()
    for field, value in settings.model_dump().items():
        if "auth_token" in field.lower() and value:
            value = str(value)[:4] + "****"
        click.echo(f"KH_{field}={value}")


@config.command("reset-batch-size")
def config_reset_batch_size():
    """Reset OOM-degraded batch size to default."""
    from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder

    settings = _get_settings()
    embedder = FlagEmbeddingEmbedder(settings)
    asyncio.run(embedder.reset_batch_size())
    click.echo(f"Batch size reset to {settings.EMBED_BATCH_SIZE}")


@cli.command()
@click.option("--host", default=None, help="Bind address.")
@click.option("--port", default=None, type=int, help="Bind port.")
def serve(host, port):
    """Start the MCP server."""
    from knowledge_hub.server.mcp_server import run_mcp_server

    settings = _get_settings()
    if host:
        settings.MCP_HOST = host
    if port:
        settings.MCP_PORT = port
    run_mcp_server(settings)


if __name__ == "__main__":
    cli()
