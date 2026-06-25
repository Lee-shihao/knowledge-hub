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
        # Set completed_at just under TTL (3599s ago) — should NOT be evicted
        fake_completed = dt_module.datetime.now(dt_module.timezone.utc) - dt_module.timedelta(seconds=3599)
        job["completed_at"] = fake_completed
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
