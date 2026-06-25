"""CLI for knowledge-hub: index, query, manage, and serve.

Heavy imports (FlagEmbedding, torch, llama_index) are deferred to command
execution time so that `kh --help` and `kh config show` respond instantly.
"""

import asyncio
import os
import warnings
from pathlib import Path

import click
import structlog

from knowledge_hub.config import Settings

# Suppress transformers tokenizer warnings (non-actionable for end users)
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

logger = structlog.get_logger()


def _get_settings() -> Settings:
    return Settings()


def _build_pipeline(settings):
    """Build ingestion pipeline (triggers heavy imports)."""
    from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
    from knowledge_hub.ingestion.loaders import DocumentLoader
    from knowledge_hub.ingestion.chunker import SemanticChunker
    from knowledge_hub.ingestion.pipeline import IngestionPipeline
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import QdrantVectorStore, build_qdrant_client

    client = build_qdrant_client(settings)
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
    from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
    from knowledge_hub.retrieval.query_engine import QueryEngine
    from knowledge_hub.retrieval.reranker import Reranker
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import QdrantVectorStore, build_qdrant_client

    client = build_qdrant_client(settings)
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

    # Expand directory to file list, or use None for default DATA_DIR
    if path:
        path_obj = Path(path)
        if path_obj.is_dir():
            paths = list(path_obj.rglob("*"))
        else:
            paths = [path_obj]
    else:
        paths = None

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
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import build_qdrant_client

    settings = _get_settings()
    client = build_qdrant_client(settings)
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
    from knowledge_hub.storage.metadata import SourceMetadataManager
    from knowledge_hub.storage.vector_store import build_qdrant_client

    settings = _get_settings()
    data_dir = Path(settings.DATA_DIR)
    local_files = {p.name for p in data_dir.rglob("*") if p.is_file()} if data_dir.exists() else set()
    client = build_qdrant_client(settings)
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
@click.option("--host", default=None, help="Bind address for MCP and upload servers.")
@click.option("--port", default=None, type=int, help="Bind port for MCP server.")
@click.option("--upload-port", default=None, type=int, help="Bind port for upload server.")
@click.option("--no-upload", is_flag=True, help="Start MCP server only (no upload).")
def serve(host, port, upload_port, no_upload):
    """Start MCP and HTTP upload servers."""
    import anyio
    import uvicorn
    from knowledge_hub.server.app_state import AppState
    from knowledge_hub.server.upload_server import create_upload_app

    settings = _get_settings()
    if host:
        settings.SERVER_HOST = host
    if port:
        settings.SERVER_PORT = port
    if upload_port:
        settings.UPLOAD_PORT = upload_port
    if not settings.UPLOAD_ENABLED:
        no_upload = True

    async def _main():
        state = await AppState.create(settings)

        if no_upload:
            config = uvicorn.Config(
                state.mcp.http_app(
                    transport="streamable-http",
                    stateless_http=True,
                    json_response=True,
                ),
                host=settings.SERVER_HOST,
                port=settings.SERVER_PORT,
            )
            logger.info(
                "server_starting",
                mcp=f"http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/mcp",
            )
            await uvicorn.Server(config).serve()
        else:
            await _run_servers(state, settings)

    anyio.run(_main)


async def _run_servers(state, settings):
    """Start MCP and upload servers in the same anyio task group."""
    import anyio
    import uvicorn
    from knowledge_hub.server.upload_server import create_upload_app

    mcp_app = state.mcp.http_app(
        transport="streamable-http",
        stateless_http=True,
        json_response=True,
    )
    upload_app = create_upload_app(state)

    mcp_config = uvicorn.Config(
        mcp_app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level="warning",
    )
    upload_config = uvicorn.Config(
        upload_app,
        host=settings.SERVER_HOST,
        port=settings.UPLOAD_PORT,
        log_level="warning",
    )

    logger.info(
        "server_starting",
        mcp=f"http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/mcp",
        upload=f"http://{settings.SERVER_HOST}:{settings.UPLOAD_PORT}/upload",
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(uvicorn.Server(mcp_config).serve)
        tg.start_soon(uvicorn.Server(upload_config).serve)

    await _shutdown(state)


async def _shutdown(state):
    """Wait for running jobs and close connections before exit."""
    if state.job_manager.has_running_job():
        logger.info("waiting_for_running_job")
        await state.job_manager.wait_until_idle(timeout=300.0)
    state.qdrant_client.close()
    logger.info("shutdown_complete")


if __name__ == "__main__":
    cli()
