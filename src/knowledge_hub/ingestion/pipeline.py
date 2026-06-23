"""IngestionPipeline — orchestrates the full ingestion flow: load → chunk → embed → store."""

import json
from pathlib import Path
from dataclasses import dataclass, field

import structlog

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import FlagEmbeddingEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager

logger = structlog.get_logger()


@dataclass
class IngestionReport:
    """Summary of an ingestion run."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    orphans_cleaned: int = 0
    failed_files: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrates the full ingestion flow: load → chunk → embed → store.

    Handles incremental updates via source_hash comparison and
    orphan vector cleanup after ingestion.
    """

    def __init__(
        self,
        settings: Settings,
        loader: DocumentLoader,
        chunker: SemanticChunker,
        embedder: FlagEmbeddingEmbedder,
        vector_store: QdrantVectorStore,
        metadata_mgr: SourceMetadataManager,
    ):
        self.settings = settings
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = vector_store
        self._metadata = metadata_mgr

    async def run(
        self,
        paths: list[Path] | None = None,
        force: bool = False,
        tags: list[str] | None = None,
    ) -> IngestionReport:
        """Run the ingestion pipeline.

        Args:
            paths: List of file paths to ingest. If None, scans DATA_DIR.
            force: If True, re-ingest even if source_hash is unchanged.
            tags: CLI-supplied tags. Sidecar .meta.json overrides these.

        Returns:
            IngestionReport summarizing the run.
        """
        tags = tags or []
        report = IngestionReport()

        if paths is None:
            data_dir = Path(self.settings.DATA_DIR)
            paths = list(data_dir.rglob("*")) if data_dir.exists() else []

        files = [p for p in paths if p.is_file()]
        local_files = {p.name for p in files}
        report.total = len(files)

        if not files:
            # Still run orphan cleanup even with no new files
            report.orphans_cleaned = await self._metadata.orphan_cleanup(local_files)
            logger.info("no_files_to_ingest")
            return report

        for file_path in files:
            try:
                source_hash = self._loader.compute_hash(file_path)
                source_name = file_path.name

                # Check for existing hash (incremental update)
                existing_hash = None
                if not force:
                    existing_hash = await self._metadata.get_hash(source_name)
                    if existing_hash == source_hash:
                        logger.debug("file_unchanged_skipped", file=source_name)
                        report.skipped += 1
                        continue

                # Remove old chunks if re-ingesting changed file
                if existing_hash is None:
                    existing_hash = await self._metadata.get_hash(source_name)
                if existing_hash:
                    await self._store.delete_by_source(source_name)

                # Load sidecar metadata
                file_tags = list(tags)
                sidecar = file_path.parent / ".meta.json"
                if sidecar.exists():
                    sidecar_data = json.loads(sidecar.read_text())
                    if "tags" in sidecar_data:
                        # Merge: sidecar tags take priority, CLI tags kept as fallback
                        sidecar_tags = set(sidecar_data["tags"])
                        file_tags = sidecar_data["tags"] + [
                            t for t in file_tags if t not in sidecar_tags
                        ]

                # Directory name as fallback tag
                dir_tag = file_path.parent.name
                if dir_tag and dir_tag not in file_tags:
                    file_tags.append(dir_tag)

                # Load → Chunk → Embed → Store
                docs = self._loader.load_files([file_path])
                if not docs:
                    continue

                chunks = self._chunker.chunk(docs, source_name, source_hash)
                if not chunks:
                    continue

                # Embed all chunks
                texts = [c.text for c in chunks]
                embeddings = await self._embedder.embed_texts(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.dense_embedding = emb["dense"]
                    chunk.sparse_embedding = emb["sparse"]
                    # Apply tags
                    chunk.metadata.tags = file_tags

                # Store
                await self._store.upsert_chunks(chunks)
                await self._metadata.upsert(source_name, source_hash, len(chunks))

                report.succeeded += 1
                logger.info("file_ingested", file=source_name, chunks=len(chunks))

            except Exception as e:
                logger.error("ingestion_failed", file=str(file_path), error=str(e))
                report.failed += 1
                report.failed_files.append(str(file_path))

        # Orphan cleanup: remove vectors for files no longer on disk
        report.orphans_cleaned = await self._metadata.orphan_cleanup(local_files)

        logger.info(
            "ingestion_complete",
            total=report.total,
            succeeded=report.succeeded,
            failed=report.failed,
            skipped=report.skipped,
            orphans=report.orphans_cleaned,
        )
        return report
