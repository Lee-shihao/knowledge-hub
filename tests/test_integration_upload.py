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
