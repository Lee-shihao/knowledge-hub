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
    return Settings(SERVER_AUTH_TOKEN="test-token-123")


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
        """When SERVER_AUTH_TOKEN is set, missing Bearer token should return 401."""
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
        assert result == "passwd"

    def test_replaces_special_chars(self):
        r"""Should replace characters outside [\w\-.] with underscore."""
        from knowledge_hub.server.upload_server import _safe_filename

        result = _safe_filename("my file (1).md")
        assert result == "my_file__1_.md"

    def test_preserves_safe_chars(self):
        """Should keep alphanumeric, dash, dot, underscore."""
        from knowledge_hub.server.upload_server import _safe_filename

        result = _safe_filename("my-file_v2.0.md")
        assert result == "my-file_v2.0.md"
