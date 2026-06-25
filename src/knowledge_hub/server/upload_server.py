"""HTTP upload server — receives file uploads and queues ingestion jobs.

Exposes POST /upload (multipart/form-data) and GET /upload/status/{job_id}.
Auth reuses KH_SERVER_AUTH_TOKEN (Bearer). Format validation reuses SUPPORTED_SUFFIXES
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
    """Check Bearer token if SERVER_AUTH_TOKEN is configured.

    Returns True if auth passes or is not required.
    """
    token = state.settings.SERVER_AUTH_TOKEN
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
