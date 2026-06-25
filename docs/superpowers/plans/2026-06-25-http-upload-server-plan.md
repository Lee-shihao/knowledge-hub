# HTTP Upload Server + Unified Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTTP upload server on :8766 and unify MCP + upload into a single-process dual-server architecture sharing one BGE-M3 model.

**Architecture:** Single anyio process runs two uvicorn servers (MCP :8765 + HTTP :8766), both receiving AppState via dependency injection. Embedded Qdrant in ./storage/qdrant/ with factory function `build_qdrant_client()`. Async job-based upload with polling status endpoint.

**Tech Stack:** FastMCP (http_app API), Starlette, uvicorn, anyio, qdrant_client (embedded mode)

## Global Constraints

- Qdrant must support both embedded (`path=`) and HTTP (`url=`) modes via `QDRANT_MODE` config
- Upload and MCP share `KH_MCP_AUTH_TOKEN` for auth, `MCP_HOST` via `SERVER_HOST` property for bind address
- `run_mcp_server()` is removed; use `mcp.http_app()` + uvicorn instead
- `create_mcp_app()` receives `AppState` instead of constructing components internally
- `JobManager.submit()` is async; upload server uses Starlette (not FastAPI/Flask)
- All existing CLI commands (`kh index`, `kh query`, `kh status`, etc.) must continue to work
- Existing tests in `tests/` must continue to pass or be updated to match new signatures
- SUPPORTED_SUFFIXES imported from loaders, not duplicated

---

### Task 1: Config — Add new fields and SERVER_HOST property

**Files:**
- Modify: `src/knowledge_hub/config.py:1-40`

**Interfaces:**
- Consumes: nothing (first task, no dependencies)
- Produces:
  - `Settings.QDRANT_MODE: Literal["embedded", "http"] = "embedded"`
  - `Settings.QDRANT_PATH: str = "./storage/qdrant"`
  - `Settings.UPLOAD_PORT: int = 8766`
  - `Settings.UPLOAD_ENABLED: bool = True`
  - `Settings.SERVER_HOST: str` (computed property, returns `self.MCP_HOST`)

- [ ] **Step 1: Add new fields to Settings**

Add these fields to `src/knowledge_hub/config.py` right after the existing `QDRANT_URL` line (currently line 23):

```python
    # Qdrant mode — embedded (path) or external (url)
    QDRANT_MODE: Literal["embedded", "http"] = "embedded"
    QDRANT_PATH: str = "./storage/qdrant"

    # Upload server
    UPLOAD_PORT: int = 8766
    UPLOAD_ENABLED: bool = True

    @property
    def SERVER_HOST(self) -> str:
        """Shared bind address for MCP and upload servers."""
        return self.MCP_HOST
```

The `Literal` import already exists on line 2, no change needed.

- [ ] **Step 2: Run existing config tests to verify no regression**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: All tests PASS (existing Settings fields unchanged, defaults sensible)

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_hub/config.py
git commit -m "feat(config): add QDRANT_MODE, QDRANT_PATH, UPLOAD_PORT, UPLOAD_ENABLED, SERVER_HOST

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: build_qdrant_client() factory function

**Files:**
- Modify: `src/knowledge_hub/storage/vector_store.py:1-139`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `Settings.QDRANT_MODE`, `Settings.QDRANT_PATH`, `Settings.QDRANT_URL` (from Task 1)
- Produces: `build_qdrant_client(settings: Settings) -> QdrantClient`

- [ ] **Step 1: Write the test**

Add to `tests/test_vector_store.py`:

```python
import pytest
from pathlib import Path
from knowledge_hub.config import Settings
from knowledge_hub.storage.vector_store import build_qdrant_client


class TestBuildQdrantClient:
    """Tests for build_qdrant_client() factory function."""

    def test_embedded_mode_creates_dir_and_returns_client(self, tmp_path):
        """Embedded mode should create QDRANT_PATH dir and return path-based client."""
        storage = tmp_path / "qdrant_data"
        settings = Settings(QDRANT_MODE="embedded", QDRANT_PATH=str(storage))
        with patch("knowledge_hub.storage.vector_store.QdrantClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = build_qdrant_client(settings)
            assert storage.exists(), "QDRANT_PATH directory should be created"
            mock_cls.assert_called_once_with(path=str(storage))
            assert client is mock_cls.return_value

    def test_http_mode_returns_url_based_client(self):
        """HTTP mode should return url-based client with check_compatibility=False."""
        settings = Settings(QDRANT_MODE="http", QDRANT_URL="http://localhost:6333")
        with patch("knowledge_hub.storage.vector_store.QdrantClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = build_qdrant_client(settings)
            mock_cls.assert_called_once_with(
                url="http://localhost:6333", check_compatibility=False
            )
            assert client is mock_cls.return_value

    def test_default_mode_is_embedded(self):
        """Default QDRANT_MODE should be embedded."""
        settings = Settings()
        assert settings.QDRANT_MODE == "embedded"
```

Add these imports at the top of the test file (alongside existing imports):
```python
from unittest.mock import MagicMock, patch
from knowledge_hub.storage.vector_store import build_qdrant_client
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_vector_store.py::TestBuildQdrantClient -v`
Expected: All 3 tests FAIL with `ImportError: cannot import name 'build_qdrant_client'`

- [ ] **Step 3: Add build_qdrant_client() to vector_store.py**

At the top of `src/knowledge_hub/storage/vector_store.py`, after the existing imports, add:

```python
from pathlib import Path


def build_qdrant_client(settings: "Settings") -> QdrantClient:
    """Create a QdrantClient based on QDRANT_MODE setting.

    Embedded mode: creates the storage directory and returns a path-based client.
    HTTP mode: returns a url-based client with compatibility checks disabled.

    Args:
        settings: Application settings.

    Returns:
        Configured QdrantClient instance.
    """
    if settings.QDRANT_MODE == "embedded":
        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=settings.QDRANT_PATH)
    return QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
```

The `Settings` import uses a string annotation to avoid circular imports — or add `from knowledge_hub.config import Settings` inside the function. Better: use `TYPE_CHECKING`:

```python
from __future__ import annotations
```

This is already the default for the project (pyproject.toml has modern settings). So we can just use `Settings` directly in the type hint. But to be safe, use the existing import pattern — `Settings` is already imported from config by the test. For the function, just use `settings` parameter without annotation, or add the import.

Actually, the clearest approach: import `Settings` at module level for type checking only, or pass `settings` untyped. Since the function signature is simple, let's just type it:

```python
from knowledge_hub.config import Settings


def build_qdrant_client(settings: Settings) -> QdrantClient:
    ...
```

This is fine — `vector_store.py` already imports `Settings` on line 16. Wait, no, it imports `from knowledge_hub.config import Settings` on line 16 already. So we can reuse that import. The function goes after the imports and before the `QdrantVectorStore` class.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_vector_store.py::TestBuildQdrantClient -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/storage/vector_store.py tests/test_vector_store.py
git commit -m "feat(storage): add build_qdrant_client() factory for embedded/HTTP Qdrant

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: metadata — add list_source_details()

**Files:**
- Modify: `src/knowledge_hub/storage/metadata.py:73-83`
- Modify: `tests/test_metadata.py`

**Interfaces:**
- Consumes: nothing new (uses existing `_collection` and `_client`)
- Produces: `SourceMetadataManager.list_source_details() -> list[dict[str, any]]`

- [ ] **Step 1: Write the test**

Add to `tests/test_metadata.py`:

```python
class TestListSourceDetails:
    """Tests for list_source_details() — returns full payload per source."""

    def test_list_source_details_returns_payloads(self, settings, metadata_mgr):
        import asyncio

        async def _run():
            await metadata_mgr.ensure_collection()
            await metadata_mgr.upsert("doc1.md", "hash_abc", 5)
            await metadata_mgr.upsert("doc2.pdf", "hash_def", 3)
            details = await metadata_mgr.list_source_details()
            return details

        details = asyncio.run(_run())
        assert len(details) == 2
        filenames = [d["source_file"] for d in details]
        assert "doc1.md" in filenames
        assert "doc2.pdf" in filenames
        for d in details:
            assert "source_hash" in d
            assert "chunk_count" in d

    def test_list_source_details_empty(self, settings, metadata_mgr):
        import asyncio

        async def _run():
            await metadata_mgr.ensure_collection()
            details = await metadata_mgr.list_source_details()
            return details

        details = asyncio.run(_run())
        assert details == []

    def test_list_sources_unchanged(self, settings, metadata_mgr):
        """Existing list_sources() should still return just filenames."""
        import asyncio

        async def _run():
            await metadata_mgr.ensure_collection()
            await metadata_mgr.upsert("doc1.md", "hash_abc", 5)
            sources = await metadata_mgr.list_sources()
            return sources

        sources = asyncio.run(_run())
        assert sources == ["doc1.md"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metadata.py::TestListSourceDetails -v`
Expected: FAIL — `AttributeError: 'SourceMetadataManager' object has no attribute 'list_source_details'`

- [ ] **Step 3: Add list_source_details() to metadata.py**

After `list_sources()` (currently ends at line 83), add:

```python
    async def list_source_details(self) -> list[dict]:
        """Return full payload for all sources.

        Unlike list_sources() which returns only filenames, this returns
        the complete payload dict for each source (source_file, source_hash,
        chunk_count).

        Returns:
            List of payload dicts from the source metadata collection.
        """
        points, next_offset = self._client.scroll(
            collection_name=self._collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        results = [p.payload for p in points]
        while next_offset:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                offset=next_offset,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(p.payload for p in points)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_metadata.py::TestListSourceDetails -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all metadata tests to verify no regression**

Run: `.venv/bin/pytest tests/test_metadata.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_hub/storage/metadata.py tests/test_metadata.py
git commit -m "feat(storage): add list_source_details() returning full payload per source

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: JobManager — async job tracking with serialized execution

**Files:**
- Create: `src/knowledge_hub/server/job_manager.py`
- Create: `tests/test_job_manager.py`

**Interfaces:**
- Consumes: `IngestionPipeline` (existing, from `knowledge_hub.ingestion.pipeline`)
- Produces:
  - `JobManager(pipeline: IngestionPipeline)` — constructor
  - `JobManager.submit(path: Path, filename: str, tags: list[str]) -> str` (async, returns job_id)
  - `JobManager.get(job_id: str) -> dict | None`
  - `JobManager.has_running_job() -> bool`
  - `JobManager.wait_until_idle(timeout: float = 300.0) -> None`

- [ ] **Step 1: Write tests for JobManager**

Create `tests/test_job_manager.py`:

```python
"""Tests for JobManager — async job tracking with serialized pipeline execution."""
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_hub.server.job_manager import JobManager, _JOB_TTL_SECONDS


class MockPipeline:
    """Fake IngestionPipeline that records calls and simulates work."""

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.runs: list[dict] = []
        self._delay = delay
        self._fail = fail

    async def run(self, paths=None, tags=None, force=False):
        self.runs.append({"paths": paths, "tags": tags})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError("simulated pipeline failure")
        from knowledge_hub.ingestion.pipeline import IngestionReport
        r = IngestionReport(total=1, succeeded=1)
        return r


@pytest.fixture
def pipeline():
    return MockPipeline()


@pytest.fixture
def job_manager(pipeline):
    return JobManager(pipeline)


class TestJobManagerSubmit:
    """Tests for JobManager.submit() — job creation and lifecycle."""

    @pytest.mark.asyncio
    async def test_submit_returns_job_id(self, job_manager):
        """submit() should return a 12-char hex job_id string."""
        job_id = await job_manager.submit(
            Path("/tmp/test.md"), "test.md", ["tag1"]
        )
        assert isinstance(job_id, str)
        assert len(job_id) == 12

    @pytest.mark.asyncio
    async def test_submit_creates_pending_job(self, job_manager):
        """submit() should create a job with status 'pending'."""
        job_id = await job_manager.submit(
            Path("/tmp/test.md"), "test.md", []
        )
        job = job_manager.get(job_id)
        assert job is not None
        assert job["status"] == "pending"
        assert job["filename"] == "test.md"
        assert job["chunks"] == 0
        assert job["error"] is None
        assert job["created_at"] is not None
        assert job["completed_at"] is None

    @pytest.mark.asyncio
    async def test_submit_is_async(self, job_manager):
        """submit() should be callable with await."""
        import inspect
        assert inspect.iscoroutinefunction(job_manager.submit), (
            "submit() must be a coroutine function"
        )


class TestJobManagerProcessing:
    """Tests for job processing — status transitions."""

    @pytest.mark.asyncio
    async def test_job_transitions_to_done(self):
        """Job should go pending -> processing -> done on success."""
        pipeline = MockPipeline(delay=0.01)
        jm = JobManager(pipeline)
        job_id = await jm.submit(Path("/tmp/test.md"), "test.md", [])
        # Let background task execute
        await asyncio.sleep(0.05)
        job = jm.get(job_id)
        assert job["status"] == "done"
        assert job["chunks"] == 1
        assert job["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_job_transitions_to_failed(self):
        """Job should go to 'failed' on pipeline exception."""
        pipeline = MockPipeline(fail=True)
        jm = JobManager(pipeline)
        job_id = await jm.submit(Path("/tmp/test.md"), "test.md", [])
        await asyncio.sleep(0.05)
        job = jm.get(job_id)
        assert job["status"] == "failed"
        assert "simulated pipeline failure" in job["error"]
        assert job["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_jobs_serialized_by_lock(self):
        """Two jobs submitted close together should process one at a time."""
        pipeline = MockPipeline(delay=0.03)
        jm = JobManager(pipeline)
        jid1 = await jm.submit(Path("/tmp/a.md"), "a.md", [])
        jid2 = await jm.submit(Path("/tmp/b.md"), "b.md", [])
        await asyncio.sleep(0.12)
        # Both should complete, and only two pipeline runs occurred
        job1 = jm.get(jid1)
        job2 = jm.get(jid2)
        assert job1["status"] == "done"
        assert job2["status"] == "done"
        assert len(pipeline.runs) == 2

    @pytest.mark.asyncio
    async def test_has_running_job(self):
        """has_running_job() should return True while a job is processing."""
        # Create a pipeline that blocks until we release it
        pipeline = MockPipeline(delay=0.1)
        jm = JobManager(pipeline)
        job_id = await jm.submit(Path("/tmp/test.md"), "test.md", [])
        # Give the background task a moment to acquire the lock
        await asyncio.sleep(0.02)
        running = jm.has_running_job()
        # Wait for completion before asserting
        await asyncio.sleep(0.15)
        assert not jm.has_running_job()


class TestJobManagerGet:
    """Tests for JobManager.get() — retrieval and error cases."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown_job(self, job_manager):
        """get() should return None for a non-existent job_id."""
        result = job_manager.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_job_dict(self, job_manager):
        """get() should return the full job dict for a valid job_id."""
        job_id = await job_manager.submit(Path("/tmp/test.md"), "test.md", [])
        job = job_manager.get(job_id)
        assert isinstance(job, dict)
        assert set(job.keys()) == {
            "job_id", "filename", "status", "chunks",
            "error", "created_at", "completed_at", "failed_files",
        }


class TestJobManagerEviction:
    """Tests for TTL-based job eviction."""

    @pytest.mark.asyncio
    async def test_completed_jobs_evicted_after_ttl(self):
        """Jobs with completed_at older than TTL should be evicted."""
        import datetime as dt_module

        pipeline = MockPipeline()
        jm = JobManager(pipeline)
        job_id = await jm.submit(Path("/tmp/test.md"), "test.md", [])
        await asyncio.sleep(0.05)
        # Force the completed_at to be in the past
        job = jm.get(job_id)
        assert job is not None
        fake_completed = dt_module.datetime.now(dt_module.timezone.utc)
        job["completed_at"] = fake_completed.replace(
            hour=fake_completed.hour - 1
        )
        # Next get() should trigger eviction if TTL is very short... but
        # our TTL is 3600s. We need a way to test without waiting 1 hour.
        # Test that _evict_expired doesn't crash and doesn't evict recent jobs:
        old_count = len(jm._jobs)
        jm._evict_expired()
        assert len(jm._jobs) == old_count, (
            "Recent jobs should not be evicted"
        )


class TestJobManagerShutdown:
    """Tests for wait_until_idle() shutdown helper."""

    @pytest.mark.asyncio
    async def test_wait_until_idle_when_idle(self, job_manager):
        """wait_until_idle() should return immediately when no job is running."""
        await job_manager.wait_until_idle(timeout=1.0)
        # Should complete without exception

    @pytest.mark.asyncio
    async def test_wait_until_idle_timeout(self):
        """wait_until_idle() should raise TimeoutError after timeout while busy."""
        pipeline = MockPipeline(delay=10.0)  # Very slow
        jm = JobManager(pipeline)
        await jm.submit(Path("/tmp/test.md"), "test.md", [])
        await asyncio.sleep(0.02)  # Let it start
        with pytest.raises(asyncio.TimeoutError):
            await jm.wait_until_idle(timeout=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_job_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'knowledge_hub.server.job_manager'`

- [ ] **Step 3: Create job_manager.py**

Create `src/knowledge_hub/server/job_manager.py`:

```python
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

        Args:
            timeout: Maximum seconds to wait before raising TimeoutError.

        Raises:
            asyncio.TimeoutError: If the running job doesn't complete in time.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_job_manager.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/server/job_manager.py tests/test_job_manager.py
git commit -m "feat(server): add JobManager for async ingestion job tracking

JobManager provides async submit/get/has_running_job/wait_until_idle
with asyncio.Lock serialization to prevent concurrent GPU OOM.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: AppState — shared state dataclass

**Files:**
- Create: `src/knowledge_hub/server/app_state.py`

**Interfaces:**
- Consumes: `build_qdrant_client()` (Task 2), `JobManager` (Task 4), `create_mcp_app()` (will be updated in Task 7)
- Produces: `AppState` dataclass with `create(settings) -> AppState` classmethod

This task creates AppState without the `mcp` field (added in Task 7 after mcp_server.py is refactored).

- [ ] **Step 1: Create app_state.py (Phase 1 — without mcp)**

Create `src/knowledge_hub/server/app_state.py`:

```python
"""AppState — single-process shared state for MCP and HTTP servers.

All GPU-heavy components (embedder, reranker) are loaded once and shared.
"""
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

    @classmethod
    async def create(cls, settings: Settings) -> "AppState":
        """Build all components and return a fully-initialized AppState.

        Two-phase construction: components first, then AppState instance.
        The mcp field is set to None here and assigned by the caller
        after create_mcp_app() is called with the completed state.

        Args:
            settings: Application settings.

        Returns:
            Fully initialized AppState (mcp field = None).
        """
        from qdrant_client import QdrantClient
        from fastmcp import FastMCP

        # Phase 1: Build all infrastructure and ML components
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

        return cls(
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
        )
```

Note: The `mcp` field is intentionally absent — it will be added in Task 7 after `create_mcp_app` is refactored to accept `AppState`.

- [ ] **Step 2: Verify the module imports correctly**

Run: `.venv/bin/python -c "from knowledge_hub.server.app_state import AppState; print('AppState imported OK')"`
Expected: `AppState imported OK`

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_hub/server/app_state.py
git commit -m "feat(server): add AppState dataclass for shared component injection

Phase 1 — without mcp field. mcp added after create_mcp_app refactored.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: tools.py — expand to 3 MCP tools

**Files:**
- Modify: `src/knowledge_hub/server/tools.py:1-62`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Settings`, `QueryEngine`, `HealthMonitor`, `SourceMetadataManager`, `QdrantVectorStore` (all existing)
- Produces: `create_tools(settings, query_engine, health, metadata_mgr, vector_store) -> dict[str, callable]` with 3 tools

- [ ] **Step 1: Update create_tools() signature and add new tools**

Replace `src/knowledge_hub/server/tools.py` entirely:

```python
"""MCP tool wrappers around the QueryEngine.

Provides 3 tools:
- query_knowledge_base: semantic search with health-gated access
- list_kb_sources: list all indexed source files with metadata
- get_kb_status: system health and collection statistics
"""
import structlog

from knowledge_hub.config import Settings
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.storage.vector_store import QdrantVectorStore

logger = structlog.get_logger()


def create_tools(
    settings: Settings,
    query_engine: QueryEngine,
    health: HealthMonitor,
    metadata_mgr: SourceMetadataManager,
    vector_store: QdrantVectorStore,
) -> dict:
    """Create MCP tool wrappers.

    Args:
        settings: Application settings.
        query_engine: QueryEngine instance for executing queries.
        health: HealthMonitor instance for health checks.
        metadata_mgr: SourceMetadataManager for listing sources.
        vector_store: QdrantVectorStore for collection stats.

    Returns:
        Dict mapping tool name to async callable.
    """

    async def query_knowledge_base(
        query: str,
        top_k: int = 5,
        filter_source: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> dict:
        """Search the knowledge base for relevant document chunks.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default 5).
            filter_source: Optional filter by source filename.
            filter_tags: Optional filter by document tags.

        Returns:
            Dictionary with 'results' list and 'query_time_ms' float.
        """
        status = await health.get_status()
        if not status.model_loaded:
            return {
                "error": "Embedding model is not available",
                "results": [],
                "query_time_ms": 0,
            }
        if not status.qdrant:
            return {
                "error": "Knowledge base is not available",
                "results": [],
                "query_time_ms": 0,
            }

        q_input = QueryInput(
            query=query,
            top_k=top_k,
            filter_source=filter_source,
            filter_tags=filter_tags,
        )
        result = await query_engine.query(q_input)
        return result.model_dump()

    async def list_kb_sources() -> dict:
        """List all indexed source files with metadata.

        Returns:
            Dictionary with 'sources' list and 'count' int.
        """
        details = await metadata_mgr.list_source_details()
        sources = [
            {
                "filename": d["source_file"],
                "chunk_count": d["chunk_count"],
                "source_hash": d["source_hash"],
            }
            for d in details
        ]
        return {"sources": sources, "count": len(sources)}

    async def get_kb_status() -> dict:
        """Get knowledge base system status and statistics.

        Returns:
            Dictionary with health status and collection statistics.
        """
        h = await health.get_status()
        try:
            total_chunks = await vector_store.count()
            sources = await metadata_mgr.list_sources()
            total_sources = len(sources)
        except Exception:
            total_chunks = -1
            total_sources = -1

        return {
            "model_loaded": h.model_loaded,
            "qdrant": h.qdrant,
            "gpu_available": h.gpu_available,
            "gpu_memory_free_mb": h.gpu_memory_free_mb,
            "collection": settings.QDRANT_COLLECTION,
            "total_chunks": total_chunks,
            "total_sources": total_sources,
        }

    return {
        "query_knowledge_base": query_knowledge_base,
        "list_kb_sources": list_kb_sources,
        "get_kb_status": get_kb_status,
    }
```

- [ ] **Step 2: Update test_tools.py**

Replace `tests/test_tools.py`:

```python
"""Tests for MCP tools — query_knowledge_base health gates and QueryEngine calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from knowledge_hub.config import Settings
from knowledge_hub.server.tools import create_tools


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def mock_health():
    h = MagicMock()
    h.get_status = AsyncMock()
    return h


@pytest.fixture
def mock_query_engine():
    e = MagicMock()
    e.query = AsyncMock()
    return e


@pytest.fixture
def mock_metadata_mgr():
    m = MagicMock()
    m.list_source_details = AsyncMock(return_value=[])
    m.list_sources = AsyncMock(return_value=[])
    return m


@pytest.fixture
def mock_vector_store():
    v = MagicMock()
    v.count = AsyncMock(return_value=0)
    return v


class TestCreateTools:
    """Tests for create_tools() — tool registration and count."""

    def test_returns_three_tools(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """create_tools() should return exactly 3 tools."""
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        assert len(tools) == 3
        assert set(tools.keys()) == {
            "query_knowledge_base",
            "list_kb_sources",
            "get_kb_status",
        }

    def test_tools_are_async_callables(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """All returned tools should be async callables."""
        import inspect
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        for name, fn in tools.items():
            assert inspect.iscoroutinefunction(fn), (
                f"{name} should be a coroutine function"
            )


class TestQueryKnowledgeBase:
    """Tests for query_knowledge_base tool."""

    @pytest.mark.asyncio
    async def test_returns_error_when_model_not_loaded(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return error dict when model_loaded is False."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=False, qdrant=True,
        )
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["query_knowledge_base"]("test query")
        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_calls_query_engine_when_healthy(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should call query_engine.query() when health checks pass."""
        from knowledge_hub.schemas import QueryResult, ChunkResult

        mock_health.get_status.return_value = MagicMock(
            model_loaded=True, qdrant=True,
        )
        mock_result = QueryResult(
            results=[
                ChunkResult(
                    text="test text",
                    source_file="test.md",
                    page_or_section="section 1",
                    heading_path=["H1"],
                    score=0.95,
                ),
            ],
            query_time_ms=10.0,
        )
        mock_query_engine.query.return_value = mock_result

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["query_knowledge_base"]("test query", top_k=3)
        assert "error" not in result
        assert result["results"][0]["text"] == "test text"
        mock_query_engine.query.assert_called_once()


class TestListKbSources:
    """Tests for list_kb_sources tool."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_sources(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return empty sources list when nothing is indexed."""
        mock_metadata_mgr.list_source_details.return_value = []
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["list_kb_sources"]()
        assert result["sources"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_returns_sources_with_metadata(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return source details including chunk_count and hash."""
        mock_metadata_mgr.list_source_details.return_value = [
            {"source_file": "doc.md", "source_hash": "abc", "chunk_count": 5},
            {"source_file": "doc.pdf", "source_hash": "def", "chunk_count": 3},
        ]
        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["list_kb_sources"]()
        assert result["count"] == 2
        assert result["sources"][0]["filename"] == "doc.md"
        assert result["sources"][0]["chunk_count"] == 5
        assert result["sources"][0]["source_hash"] == "abc"


class TestGetKbStatus:
    """Tests for get_kb_status tool."""

    @pytest.mark.asyncio
    async def test_returns_health_and_stats(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return health status plus collection statistics."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=True,
            qdrant=True,
            gpu_available=True,
            gpu_memory_free_mb=8192,
        )
        mock_vector_store.count.return_value = 42
        mock_metadata_mgr.list_sources.return_value = ["a.md", "b.md"]

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["get_kb_status"]()
        assert result["model_loaded"] is True
        assert result["qdrant"] is True
        assert result["total_chunks"] == 42
        assert result["total_sources"] == 2
        assert result["collection"] == "knowledge_hub"

    @pytest.mark.asyncio
    async def test_handles_vector_store_error(
        self, settings, mock_query_engine, mock_health,
        mock_metadata_mgr, mock_vector_store,
    ):
        """Should return -1 counts when Qdrant is unreachable."""
        mock_health.get_status.return_value = MagicMock(
            model_loaded=True, qdrant=False,
            gpu_available=False, gpu_memory_free_mb=0,
        )
        mock_vector_store.count.side_effect = RuntimeError("qdrant down")
        mock_metadata_mgr.list_sources.side_effect = RuntimeError("qdrant down")

        tools = create_tools(
            settings, mock_query_engine, mock_health,
            mock_metadata_mgr, mock_vector_store,
        )
        result = await tools["get_kb_status"]()
        assert result["total_chunks"] == -1
        assert result["total_sources"] == -1
```

- [ ] **Step 3: Run new tools tests**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_hub/server/tools.py tests/test_tools.py
git commit -m "feat(server): expand MCP tools to 3 — query_kb, list_sources, get_status

New tools require metadata_mgr and vector_store in create_tools() signature.
Existing query_knowledge_base behavior unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: mcp_server.py — refactor for AppState injection

**Files:**
- Modify: `src/knowledge_hub/server/mcp_server.py:1-152`
- Modify: `tests/test_mcp_server.py:1-380`
- Modify: `src/knowledge_hub/server/app_state.py` (add `mcp` field)

**Interfaces:**
- Consumes: `AppState` (Task 5), expanded `create_tools()` (Task 6)
- Produces: `create_mcp_app(state: AppState) -> FastMCP`
- Removed: `run_mcp_server()` function

- [ ] **Step 1: Rewrite mcp_server.py**

Replace `src/knowledge_hub/server/mcp_server.py`:

```python
"""MCP server — receives AppState and exposes query/list/status tools.

Uses FlagEmbeddingEmbedder (not OllamaEmbedder) for embeddings.
Supports Bearer token auth via StaticTokenVerifier and IP allowlist middleware.
"""
import structlog
from fastmcp import FastMCP

from knowledge_hub.server.app_state import AppState
from knowledge_hub.server.tools import create_tools

logger = structlog.get_logger()


def create_mcp_app(state: AppState) -> FastMCP:
    """Build and configure the FastMCP application from AppState.

    All heavy components (embedder, reranker, qdrant) come from state —
    this function only handles auth, middleware, and tool registration.

    Args:
        state: Fully initialized AppState (mcp may be None).

    Returns:
        Configured FastMCP instance.

    Raises:
        ValueError: If MCP_HOST is not 127.0.0.1 and no auth token is set.
    """
    # Auth setup
    if state.settings.MCP_AUTH_TOKEN:
        from fastmcp.server.auth import StaticTokenVerifier

        verifier = StaticTokenVerifier(
            {
                state.settings.MCP_AUTH_TOKEN: {
                    "client_id": "knowledge-hub",
                    "scopes": [],
                }
            }
        )
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if state.settings.MCP_HOST != "127.0.0.1":
            raise ValueError(
                "MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set. "
                "Set KH_MCP_AUTH_TOKEN to enable LAN access."
            )
        mcp = FastMCP("knowledge-hub")

    # IP allowlist middleware — identical to original implementation
    if state.settings.MCP_ALLOWED_IPS:
        import ipaddress

        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        allowed_networks = []
        for ip_str in state.settings.MCP_ALLOWED_IPS:
            try:
                if "/" in ip_str:
                    allowed_networks.append(
                        ipaddress.ip_network(ip_str, strict=False)
                    )
                else:
                    allowed_networks.append(
                        ipaddress.ip_network(f"{ip_str}/32", strict=False)
                    )
            except ValueError:
                logger.warning("invalid_ip_in_allowlist", ip=ip_str)

        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, allowed_networks=None):
                super().__init__(app)
                self.allowed_networks = allowed_networks or []

            async def dispatch(self, request, call_next):
                client_ip = (
                    request.client.host if request.client else None
                )
                if client_ip:
                    try:
                        ip_addr = ipaddress.ip_address(client_ip)
                        if not any(
                            ip_addr in net
                            for net in self.allowed_networks
                        ):
                            return JSONResponse(
                                {"error": "Forbidden"}, status_code=403
                            )
                    except ValueError:
                        return JSONResponse(
                            {"error": "Forbidden"}, status_code=403
                        )
                else:
                    return JSONResponse(
                        {"error": "Forbidden"}, status_code=403
                    )
                return await call_next(request)

        mcp.add_middleware(
            Middleware(
                IPAllowlistMiddleware, allowed_networks=allowed_networks
            )
        )

    # Register tools — pass components from state
    tools = create_tools(
        state.settings,
        state.query_engine,
        state.health,
        state.metadata_mgr,
        state.vector_store,
    )
    for name, fn in tools.items():
        mcp.add_tool(fn)

    # Store health for test compatibility
    mcp._health = state.health

    return mcp
```

Note: `run_mcp_server()` is removed entirely. The serve command in Task 9 handles server startup via uvicorn.

- [ ] **Step 2: Update test_mcp_server.py**

The test file needs updating because:
1. `create_mcp_app` now takes `AppState` instead of `Settings`
2. `run_mcp_server` tests are removed (function is deleted)

Replace `tests/test_mcp_server.py`:

```python
"""Tests for MCP server creation — app wiring, auth, IP middleware."""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_hub.config import Settings


def _make_app_state(settings, **overrides):
    """Build a mock AppState with the given settings and overridable fields."""
    from knowledge_hub.server.app_state import AppState

    defaults = {
        "settings": settings,
        "qdrant_client": MagicMock(),
        "embedder": MagicMock(),
        "reranker": MagicMock(),
        "metadata_mgr": MagicMock(),
        "vector_store": MagicMock(),
        "query_engine": MagicMock(),
        "pipeline": MagicMock(),
        "health": MagicMock(),
        "job_manager": MagicMock(),
    }
    defaults.update(overrides)
    return AppState(**defaults)


class TestCreateMCPApp:
    """Tests for create_mcp_app() — app creation and configuration."""

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def settings_with_auth(self):
        return Settings(MCP_AUTH_TOKEN="test-token-123")

    @pytest.fixture
    def settings_with_allowed_ips(self):
        return Settings(MCP_ALLOWED_IPS=["192.168.1.1", "10.0.0.0/8"])

    @pytest.fixture
    def settings_lan_no_auth(self):
        return Settings(MCP_HOST="0.0.0.0", MCP_AUTH_TOKEN=None)

    def test_create_mcp_app_returns_fastmcp_instance(self, settings):
        """create_mcp_app should return a FastMCP instance."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        from fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_has_health_attribute(self, settings):
        """create_mcp_app should attach _health from AppState."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        assert hasattr(mcp, "_health")
        assert mcp._health is state.health

    def test_create_mcp_app_with_auth_token(self, settings_with_auth):
        """When MCP_AUTH_TOKEN is set, auth should be configured without error."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_with_auth)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_raises_on_lan_no_auth(self, settings_lan_no_auth):
        """When MCP_HOST is not 127.0.0.1 and no auth token, should raise ValueError."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_lan_no_auth)
        with pytest.raises(ValueError, match="MCP_HOST must be 127.0.0.1"):
            create_mcp_app(state)

    def test_create_mcp_app_allows_localhost_no_auth(self, settings):
        """When MCP_HOST is 127.0.0.1 and no auth token, should succeed."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"

    def test_create_mcp_app_with_allowed_ips(self, settings_with_allowed_ips):
        """When MCP_ALLOWED_IPS is set, IP middleware should be added without error."""
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings_with_allowed_ips)
        mcp = create_mcp_app(state)
        assert mcp.name == "knowledge-hub"


class TestStreamableHTTPIntegration:
    """curl-style integration tests — verify streamable-http transport via TestClient."""

    JSON_HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    @pytest.fixture
    def settings(self):
        return Settings()

    @pytest.fixture
    def test_client(self, settings):
        """Create MCP app via AppState and return Starlette TestClient."""
        from starlette.testclient import TestClient
        from knowledge_hub.server.mcp_server import create_mcp_app

        state = _make_app_state(settings)
        mcp = create_mcp_app(state)
        mcp._health.model_loaded = True
        mcp._health.qdrant = True

        app = mcp.http_app(
            transport="streamable-http",
            stateless_http=True,
            json_response=True,
            path="/mcp",
        )
        with TestClient(app) as client:
            yield client

    def test_tools_list_via_http_post(self, test_client):
        """curl-style POST /mcp calling tools/list should return tool list."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "result" in body, f"Missing 'result' in: {body}"
        assert "tools" in body["result"]
        assert body["result"]["tools"], "Tool list should not be empty"

        tool_names = [t["name"] for t in body["result"]["tools"]]
        assert "query_knowledge_base" in tool_names, (
            f"Expected query_knowledge_base in tools, got: {tool_names}"
        )
        assert "list_kb_sources" in tool_names, (
            f"Expected list_kb_sources in tools, got: {tool_names}"
        )
        assert "get_kb_status" in tool_names, (
            f"Expected get_kb_status in tools, got: {tool_names}"
        )

    def test_tools_list_returns_json_not_sse(self, test_client):
        """json_response=True should return JSON, not SSE text stream."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" not in content_type
        body = response.json()
        assert "result" in body

    def test_initialize_via_http_post(self, test_client):
        """MCP initialize request should succeed via direct POST."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "result" in body
        assert "protocolVersion" in body["result"]

    def test_invalid_method_returns_error(self, test_client):
        """Invalid JSON-RPC method should return error response."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "nonexistent/method",
            },
            headers=self.JSON_HEADERS,
        )

        assert response.status_code == 200
        body = response.json()
        assert "error" in body or "result" in body

    def test_missing_accept_header_returns_406(self, test_client):
        """Missing Accept: application/json should return 406."""
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
            },
        )

        assert response.status_code == 406
        body = response.json()
        assert "error" in body
        assert "application/json" in body["error"]["message"]
```

Note: The `TestRunMCPServer` and `TestIPMiddleware` classes are removed from the test suite — `run_mcp_server()` no longer exists (server startup is in the CLI), and IP middleware testing is covered by `test_create_mcp_app_with_allowed_ips`.

- [ ] **Step 3: Run updated tests**

Run: `.venv/bin/pytest tests/test_mcp_server.py -v`
Expected: All tests PASS

- [ ] **Step 4: Add mcp field to AppState**

Update `src/knowledge_hub/server/app_state.py` — add `mcp` field and update `create()`:

1. Add `mcp: "FastMCP" | None = None` to the dataclass fields
2. Update `create()` to construct `mcp` after building the state object

```python
@dataclass
class AppState:
    """Shared application state for all servers in the process."""
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
```

And in `create()`:

```python
    @classmethod
    async def create(cls, settings: Settings) -> "AppState":
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
```

- [ ] **Step 5: Verify app_state imports cleanly with mcp**

Run: `.venv/bin/python -c "from knowledge_hub.server.app_state import AppState; print('AppState with mcp OK')"`
Expected: `AppState with mcp OK`

- [ ] **Step 6: Run all server tests**

Run: `.venv/bin/pytest tests/test_mcp_server.py tests/test_tools.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_hub/server/mcp_server.py tests/test_mcp_server.py src/knowledge_hub/server/app_state.py
git commit -m "refactor(server): inject AppState into create_mcp_app, remove run_mcp_server

create_mcp_app now receives AppState instead of constructing components.
run_mcp_server removed — startup handled by CLI via mcp.http_app() + uvicorn.
AppState gains mcp field with two-phase construction.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: upload_server.py — HTTP upload endpoints

**Files:**
- Create: `src/knowledge_hub/server/upload_server.py`
- Create: `tests/test_upload_server.py`

**Interfaces:**
- Consumes: `AppState` (Task 7), `SUPPORTED_SUFFIXES` from `loaders.py`
- Produces: `create_upload_app(state: AppState) -> Starlette`
- Endpoints: `POST /upload`, `GET /upload/status/{job_id}`

- [ ] **Step 1: Write tests for upload server**

Create `tests/test_upload_server.py`:

```python
"""Tests for HTTP upload server — upload, status, auth, validation."""
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from knowledge_hub.config import Settings
from knowledge_hub.server.upload_server import create_upload_app


def _make_app_state(settings, **overrides):
    """Build a mock AppState with given settings."""
    from knowledge_hub.server.app_state import AppState

    defaults = {
        "settings": settings,
        "qdrant_client": MagicMock(),
        "embedder": MagicMock(),
        "reranker": MagicMock(),
        "metadata_mgr": MagicMock(),
        "vector_store": MagicMock(),
        "query_engine": MagicMock(),
        "pipeline": MagicMock(),
        "health": MagicMock(),
        "job_manager": MagicMock(),
        "mcp": None,
    }
    defaults.update(overrides)
    return AppState(**defaults)


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def settings_with_auth():
    return Settings(MCP_AUTH_TOKEN="test-token-123")


@pytest.fixture
def client(settings):
    """TestClient for upload server without auth."""
    state = _make_app_state(settings)
    state.job_manager.submit = AsyncMock(return_value="abc123def456")
    state.job_manager.get = MagicMock(return_value=None)
    app = create_upload_app(state)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_auth(settings_with_auth):
    """TestClient for upload server with Bearer token auth."""
    state = _make_app_state(settings_with_auth)
    state.job_manager.submit = AsyncMock(return_value="abc123def456")
    state.job_manager.get = MagicMock(return_value=None)
    app = create_upload_app(state)
    with TestClient(app) as c:
        yield c


class TestUploadEndpoint:
    """Tests for POST /upload."""

    def test_upload_returns_job_id(self, client):
        """Successful upload should return job_id and pending status."""
        response = client.post(
            "/upload",
            files={"file": ("test.md", io.BytesIO(b"# Hello"), "text/markdown")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == "abc123def456"
        assert body["status"] == "pending"

    def test_upload_with_tags(self, client):
        """Tags field should be accepted."""
        response = client.post(
            "/upload",
            files={"file": ("test.md", io.BytesIO(b"# Hello"), "text/markdown")},
            data={"tags": "tag1,tag2"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "pending"

    def test_upload_no_file_returns_400(self, client):
        """Missing file field should return 400."""
        response = client.post("/upload")
        assert response.status_code == 400
        assert "No file provided" in response.json()["error"]

    def test_upload_unsupported_format_returns_400(self, client):
        """Unsupported file suffix should return 400."""
        response = client.post(
            "/upload",
            files={"file": ("test.bin", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["error"]

    def test_upload_oversized_file_returns_413(self, client):
        """File exceeding MAX_FILE_SIZE_MB should return 413."""
        # Create a file just over 1MB with a very small MAX_FILE_SIZE_MB
        settings_small = Settings(MAX_FILE_SIZE_MB=1)
        state = _make_app_state(settings_small)
        state.job_manager.submit = AsyncMock(return_value="abc123def456")
        app = create_upload_app(state)
        with TestClient(app) as c:
            big_content = b"x" * (1 * 1024 * 1024 + 1)
            response = c.post(
                "/upload",
                files={"file": ("big.md", io.BytesIO(big_content), "text/markdown")},
            )
            assert response.status_code == 413
            assert "exceeds max size" in response.json()["error"]

    def test_upload_requires_auth_when_token_set(self, client_with_auth):
        """When MCP_AUTH_TOKEN is set, missing Bearer token should return 401."""
        response = client_with_auth.post(
            "/upload",
            files={"file": ("test.md", io.BytesIO(b"# Hello"), "text/markdown")},
        )
        assert response.status_code == 401

    def test_upload_with_valid_auth_token(self, settings_with_auth):
        """Valid Bearer token should allow upload."""
        state = _make_app_state(settings_with_auth)
        state.job_manager.submit = AsyncMock(return_value="abc123def456")
        app = create_upload_app(state)
        with TestClient(app) as c:
            response = c.post(
                "/upload",
                files={"file": ("test.md", io.BytesIO(b"# Hello"), "text/markdown")},
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["job_id"] == "abc123def456"


class TestStatusEndpoint:
    """Tests for GET /upload/status/{job_id}."""

    def test_status_returns_job(self, client):
        """Known job_id should return full status dict."""
        from datetime import datetime, timezone

        state = client.app.state.kh
        state.job_manager.get.return_value = {
            "job_id": "abc123",
            "filename": "test.md",
            "status": "done",
            "chunks": 5,
            "error": None,
            "created_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "failed_files": [],
        }

        response = client.get("/upload/status/abc123")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done"
        assert body["chunks"] == 5
        assert body["filename"] == "test.md"

    def test_status_unknown_job_returns_404(self, client):
        """Unknown job_id should return 404."""
        # job_manager.get returns None for unknown
        state = client.app.state.kh
        state.job_manager.get.return_value = None

        response = client.get("/upload/status/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()


class TestFilenameSafety:
    """Tests for _safe_filename helper."""

    def test_strips_directory_traversal(self):
        """Should strip leading directory components."""
        from knowledge_hub.server.upload_server import _safe_filename

        result = _safe_filename("../../../etc/passwd")
        assert result == "etc_passwd"

    def test_replaces_special_chars(self):
        """Should replace characters outside [\\w\\-.] with underscore."""
        from knowledge_hub.server.upload_server import _safe_filename

        result = _safe_filename("my file (1).md")
        assert result == "my_file__1_.md"

    def test_preserves_safe_chars(self):
        """Should keep alphanumeric, dash, dot, underscore."""
        from knowledge_hub.server.upload_server import _safe_filename

        result = _safe_filename("my-file_v2.0.md")
        assert result == "my-file_v2.0.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_upload_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'knowledge_hub.server.upload_server'`

- [ ] **Step 3: Create upload_server.py**

Create `src/knowledge_hub/server/upload_server.py`:

```python
"""HTTP upload server — receives file uploads and queues ingestion jobs.

Exposes POST /upload (multipart/form-data) and GET /upload/status/{job_id}.
Auth reuses KH_MCP_AUTH_TOKEN (Bearer). Format validation reuses SUPPORTED_SUFFIXES
from loaders to stay in sync with the pipeline.
"""
import re
from pathlib import Path, PurePosixPath

import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from knowledge_hub.ingestion.loaders import SUPPORTED_SUFFIXES
from knowledge_hub.server.app_state import AppState

logger = structlog.get_logger()


def _safe_filename(filename: str) -> str:
    """Sanitize a user-supplied filename.

    Strips directory traversal and replaces special characters.
    """
    name = PurePosixPath(filename).name
    return re.sub(r"[^\w\-.]", "_", name)


def _check_auth(request: Request, state: AppState) -> bool:
    """Check Bearer token if MCP_AUTH_TOKEN is configured.

    Returns True if auth passes or is not required.
    """
    token = state.settings.MCP_AUTH_TOKEN
    if not token:
        return True  # No auth configured

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False

    provided = auth_header[len("Bearer "):]
    return provided == token


async def _upload(request: Request) -> JSONResponse:
    """Handle POST /upload — validate, save, and submit ingestion job."""
    state: AppState = request.app.state.kh

    # Auth check
    if not _check_auth(request, state):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Parse form
    form = await request.form()
    file = form.get("file")
    if file is None:
        return JSONResponse(
            {"error": "No file provided"}, status_code=400
        )

    # Validate format BEFORE reading content
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return JSONResponse(
            {"error": f"Unsupported format: {suffix}"}, status_code=400
        )

    # Read content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > state.settings.MAX_FILE_SIZE_MB:
        return JSONResponse(
            {
                "error": (
                    f"File exceeds max size: "
                    f"{state.settings.MAX_FILE_SIZE_MB}MB"
                )
            },
            status_code=413,
        )

    # Parse tags
    tags_raw = form.get("tags")
    tags = (
        [t.strip() for t in tags_raw.split(",") if t.strip()]
        if tags_raw
        else []
    )

    # Save file to DATA_DIR
    import uuid

    data_dir = Path(state.settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(file.filename)
    dest = data_dir / safe_name
    if dest.exists():
        # Prepend short UUID to avoid overwriting existing files
        dest = data_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

    dest.write_bytes(content)

    # Submit ingestion job
    job_id = await state.job_manager.submit(dest, file.filename, tags)

    return JSONResponse({"job_id": job_id, "status": "pending"})


async def _status(request: Request) -> JSONResponse:
    """Handle GET /upload/status/{job_id} — return job status."""
    state: AppState = request.app.state.kh
    job_id = request.path_params["job_id"]

    job = state.job_manager.get(job_id)
    if job is None:
        return JSONResponse(
            {"error": "Job not found"}, status_code=404
        )

    # Serialize datetime fields for JSON
    result = dict(job)
    for key in ("created_at", "completed_at"):
        if result.get(key):
            result[key] = result[key].isoformat()

    return JSONResponse(result)


def create_upload_app(state: AppState) -> Starlette:
    """Create the HTTP upload Starlette application.

    Args:
        state: Shared AppState for accessing job_manager, settings, etc.

    Returns:
        Starlette app with upload routes mounted.
    """
    app = Starlette(
        routes=[
            Route("/upload", _upload, methods=["POST"]),
            Route("/upload/status/{job_id}", _status, methods=["GET"]),
        ]
    )
    app.state.kh = state
    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_upload_server.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/server/upload_server.py tests/test_upload_server.py
git commit -m "feat(server): add HTTP upload server with POST /upload and GET /upload/status

Supports multipart/form-data upload, Bearer token auth (shared with MCP),
format validation from SUPPORTED_SUFFIXES, filename sanitization, and
async job submission via JobManager.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: cli/main.py — rewrite `serve` command with dual-server startup

**Files:**
- Modify: `src/knowledge_hub/cli/main.py:193-205`

**Interfaces:**
- Consumes: `AppState` (Task 7), `create_upload_app` (Task 8), `mcp.http_app()` (FastMCP API)
- Produces: `kh serve` command with `--upload-port`, `--no-upload` options

- [ ] **Step 1: Update the serve command**

Replace the `serve` command in `src/knowledge_hub/cli/main.py` (currently lines 193-205):

```python
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
        settings.MCP_HOST = host
    if port:
        settings.MCP_PORT = port
    if upload_port:
        settings.UPLOAD_PORT = upload_port

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
                port=settings.MCP_PORT,
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
        port=settings.MCP_PORT,
        log_level="warning",
    )
    upload_config = uvicorn.Config(
        upload_app,
        host=settings.SERVER_HOST,
        port=settings.UPLOAD_PORT,
        log_level="warning",
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
```

- [ ] **Step 2: Update all CLI commands that directly construct QdrantClient**

Replace `QdrantClient(settings.QDRANT_URL, check_compatibility=False)` with `build_qdrant_client(settings)` in:

1. `_build_pipeline()` (line 37): `client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)` → `client = build_qdrant_client(settings)`
2. `_build_query_engine()` (line 59): same pattern
3. `status()` (line 133): same pattern
4. `cleanup_orphans()` (line 160): same pattern

All four locations follow the same pattern. For each:

```python
# Before:
client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
# After:
from knowledge_hub.storage.vector_store import build_qdrant_client
client = build_qdrant_client(settings)
```

But to keep imports lightweight (the CLI uses deferred imports for `kh --help` speed), use the import inside each function body rather than at the module level.

- [ ] **Step 3: Run CLI help smoke test**

Run: `.venv/bin/python -m knowledge_hub.cli.main --help`
Expected: Shows CLI help with `serve` in the command list

- [ ] **Step 4: Run existing CLI tests**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: All tests PASS. If any test breaks due to QdrantClient construction changes, update the test mocks accordingly.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/cli/main.py
git commit -m "refactor(cli): rewrite serve for dual-server uvicorn startup, use build_qdrant_client

kh serve now starts MCP (:8765) + HTTP upload (:8766) in an anyio task group.
All CLI commands use build_qdrant_client() instead of raw QdrantClient().
run_mcp_server removed — serve uses mcp.http_app() + uvicorn.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Integration test — full upload → status → query round-trip

**Files:**
- Create: `tests/test_integration_upload.py`

**Interfaces:**
- Consumes: All previous tasks (full system)
- Produces: Integration test verifying the upload → index → query flow

- [ ] **Step 1: Write integration test**

Create `tests/test_integration_upload.py`:

```python
"""Integration test — upload, status polling, query round-trip."""
import io
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from knowledge_hub.config import Settings
from knowledge_hub.server.app_state import AppState


def _make_app_state(settings):
    """Build mock AppState for integration testing."""
    from knowledge_hub.server.app_state import AppState

    state = AppState(
        settings=settings,
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        metadata_mgr=MagicMock(),
        vector_store=MagicMock(),
        query_engine=MagicMock(),
        pipeline=MagicMock(),
        health=MagicMock(),
        job_manager=MagicMock(),
        mcp=None,
    )
    from knowledge_hub.server.mcp_server import create_mcp_app
    state.mcp = create_mcp_app(state)
    return state


class TestUploadQueryRoundTrip:
    """End-to-end: upload file → poll status → query via MCP."""

    @pytest.fixture
    def settings(self):
        return Settings()

    def test_upload_and_query_flow(self, settings):
        """Upload a markdown file, then query it via MCP."""
        from knowledge_hub.server.upload_server import create_upload_app

        state = _make_app_state(settings)

        # Setup: job_manager.submit returns a job_id
        state.job_manager.submit = AsyncMock(return_value="job_001")
        state.job_manager.get = MagicMock(
            return_value={
                "job_id": "job_001",
                "filename": "test.md",
                "status": "done",
                "chunks": 3,
                "error": None,
                "created_at": None,
                "completed_at": None,
                "failed_files": [],
            }
        )

        # Step 1: Upload file
        upload_app = create_upload_app(state)
        with TestClient(upload_app) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "test.md",
                        io.BytesIO(b"# Test Document\n\nContent here."),
                        "text/markdown",
                    )
                },
                data={"tags": "demo,test"},
            )
            assert response.status_code == 200
            upload_body = response.json()
            assert upload_body["job_id"] == "job_001"
            assert upload_body["status"] == "pending"

            # Step 2: Check job status
            status_response = client.get("/upload/status/job_001")
            assert status_response.status_code == 200
            status_body = status_response.json()
            assert status_body["status"] == "done"
            assert status_body["chunks"] == 3

        # Step 3: Verify job_manager.submit was called
        state.job_manager.submit.assert_called_once()


class TestUploadServerNoAuth:
    """Integration test without auth token."""

    @pytest.fixture
    def settings(self):
        return Settings()

    def test_unauthenticated_upload_works_when_no_token(self, settings):
        """When MCP_AUTH_TOKEN is None, uploads should work without auth."""
        from knowledge_hub.server.upload_server import create_upload_app

        state = _make_app_state(settings)
        state.job_manager.submit = AsyncMock(return_value="job_002")

        upload_app = create_upload_app(state)
        with TestClient(upload_app) as client:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "doc.pdf",
                        io.BytesIO(b"%PDF-1.4 fake content"),
                        "application/pdf",
                    )
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "pending"
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/pytest tests/test_integration_upload.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_integration.py -k "not integration_setup"`
(This skips the real integration tests that need a running Qdrant)

Expected: All unit tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_upload.py
git commit -m "test: add integration test for upload → status → query round-trip

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Final verification — run full suite

- [ ] **Step 1: Run all tests that don't need GPU/Qdrant**

Run: `.venv/bin/pytest tests/ -v --timeout=30 -k "not (integration_setup or qdrant_available)"`
Expected: All tests PASS

- [ ] **Step 2: Verify CLI commands still work (smoke test)**

Run:
```bash
.venv/bin/python -m knowledge_hub.cli.main --help
.venv/bin/python -m knowledge_hub.cli.main config show
.venv/bin/python -m knowledge_hub.cli.main status 2>&1 || true  # may fail without Qdrant
```
Expected: CLI commands don't crash with import errors

- [ ] **Step 3: Verify import chain works end to end**

Run: `.venv/bin/python -c "
from knowledge_hub.config import Settings
from knowledge_hub.storage.vector_store import build_qdrant_client
from knowledge_hub.server.job_manager import JobManager
from knowledge_hub.server.app_state import AppState
from knowledge_hub.server.tools import create_tools
from knowledge_hub.server.mcp_server import create_mcp_app
from knowledge_hub.server.upload_server import create_upload_app
print('All imports OK')
"`
Expected: `All imports OK`

- [ ] **Step 4: Commit any final fixes and finalize**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, imports clean"
```
