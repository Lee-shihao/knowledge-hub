"""JobManager — async job tracking for ingestion pipeline execution.

Manages a dict of JobStatus entries with asyncio.Lock-based serialization
to prevent concurrent embedding workloads from causing OOM on GPU.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from knowledge_hub.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger()

_JOB_TTL_SECONDS = 3600  # Completed jobs evicted after 1 hour


class JobManager:
    """Tracks async ingestion jobs with serialized execution.

    Jobs are submitted via submit() which creates a background asyncio task.
    Execution is serialized via asyncio.Lock to avoid concurrent GPU OOM.
    Completed jobs are lazily evicted after _JOB_TTL_SECONDS.
    """

    def __init__(self, pipeline: IngestionPipeline):
        self._pipeline = pipeline
        self._jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    # ---- Public API ----

    async def submit(
        self, path: Path, filename: str, tags: list[str]
    ) -> str:
        """Submit an ingestion job. Must be called from an async context.

        Args:
            path: Absolute path to the uploaded file in DATA_DIR.
            filename: Original filename (for display).
            tags: List of tags to apply to all chunks.

        Returns:
            12-character hex job_id for status polling.
        """
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        self._jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "chunks": 0,
            "error": None,
            "created_at": now,
            "completed_at": None,
            "failed_files": [],
        }
        asyncio.create_task(self._run(job_id, path, tags))
        self._evict_expired()
        return job_id

    def get(self, job_id: str) -> dict | None:
        """Get job status by ID. Returns None if not found or evicted."""
        self._evict_expired()
        return self._jobs.get(job_id)

    def has_running_job(self) -> bool:
        """Check if a job is currently being processed."""
        return self._lock.locked()

    async def wait_until_idle(self, timeout: float = 300.0):
        """Wait for any running job to complete.

        Logs a warning and returns if the timeout expires — the caller
        should proceed with shutdown (close connections, exit).

        Args:
            timeout: Maximum seconds to wait.
        """
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    pass  # Acquiring the lock means no job is running
        except asyncio.TimeoutError:
            logger.warning("shutdown_timeout_force_exit")

    # ---- Internal ----

    async def _run(self, job_id: str, path: Path, tags: list[str]):
        """Execute ingestion pipeline in background, update job status."""
        self._jobs[job_id]["status"] = "processing"
        async with self._lock:
            try:
                report = await self._pipeline.run(paths=[path], tags=tags)
                self._jobs[job_id].update(
                    status="done" if not report.failed else "failed",
                    chunks=report.succeeded,
                    error=(
                        report.failed_files[0]
                        if report.failed_files
                        else None
                    ),
                    completed_at=datetime.now(timezone.utc),
                    failed_files=report.failed_files,
                )
                # File is intentionally kept in DATA_DIR after indexing.
                # Supports re-indexing (kh index --force) and orphan cleanup.
            except Exception as e:
                self._jobs[job_id].update(
                    status="failed",
                    error=str(e),
                    completed_at=datetime.now(timezone.utc),
                )

    def _evict_expired(self):
        """Remove completed jobs older than _JOB_TTL_SECONDS."""
        now = datetime.now(timezone.utc)
        expired = [
            jid
            for jid, j in self._jobs.items()
            if j.get("completed_at")
            and (now - j["completed_at"]).total_seconds() > _JOB_TTL_SECONDS
        ]
        for jid in expired:
            del self._jobs[jid]
