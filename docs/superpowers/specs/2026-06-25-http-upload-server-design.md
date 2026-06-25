# HTTP Upload Server + Unified Architecture Design

> 2026-06-25 · Status: approved

## Overview

Add an HTTP upload server to knowledge-hub, enabling external agents (Claude Code, Herd, Hermes, etc.) to upload files into the knowledge base over HTTP, while retaining MCP-based knowledge retrieval. Both services run in a single process on a GPU server, sharing one copy of the embedding model (BGE-M3).

## Architecture

```
GPU Server — single process via kh serve
┌─────────────────────────────────────────────────────────┐
│  anyio task group                                       │
│                                                         │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ uvicorn :8765     │  │ uvicorn :8766              │  │
│  │ mcp.http_app()    │  │ Starlette upload_app       │  │
│  │                   │  │                            │  │
│  │ query_kb          │  │ POST /upload               │  │
│  │ list_sources      │  │ GET  /upload/status/{id}   │  │
│  │ get_status        │  │                            │  │
│  └────────┬──────────┘  └─────────────┬──────────────┘  │
│           │                           │                 │
│           └───────────┬───────────────┘                 │
│                       ▼                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ AppState (shared)                                │   │
│  │  - embedder (BGE-M3, GPU, single copy)           │   │
│  │  - reranker, query_engine                        │   │
│  │  - pipeline, job_manager                         │   │
│  │  - health monitor                                │   │
│  │  - qdrant_client (embedded ./storage/qdrant/)    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Process model | Single process, two uvicorn servers | GPU memory: one BGE-M3 (~2.2GB) instead of two |
| Qdrant mode | Embedded by default, HTTP fallback | Simplest single-machine deployment; configurable |
| Upload mode | Async job (submit → poll) | Avoid HTTP timeouts on large files |
| Auth | Shared `KH_MCP_AUTH_TOKEN` | Single credential for trusted LAN |
| File storage | Persist to `DATA_DIR` | Consistent with existing `kh index` flow |
| Request format | `multipart/form-data` | Universal, easy for agents to construct |
| Skill distribution | In-repo `skills/` directory | Internal project; no need for separate distribution |

## File Changes

| File | Change |
|------|--------|
| `src/knowledge_hub/config.py` | Add 4 fields: QDRANT_MODE, QDRANT_PATH, UPLOAD_PORT, UPLOAD_ENABLED |
| `src/knowledge_hub/storage/vector_store.py` | Add `build_qdrant_client()` factory function |
| `src/knowledge_hub/server/app_state.py` | **New** — AppState dataclass + JobManager |
| `src/knowledge_hub/server/upload_server.py` | **New** — Starlette app with POST /upload, GET /upload/status/{id} |
| `src/knowledge_hub/server/mcp_server.py` | Refactor to accept AppState injection instead of constructing components |
| `src/knowledge_hub/server/tools.py` | Expand to 3 tools (query_kb, list_sources, get_status) |
| `src/knowledge_hub/storage/metadata.py` | Add `list_source_details()` returning full payload |
| `src/knowledge_hub/cli/main.py` | Rewrite `serve` command with anyio + uvicorn dual-server startup |

## 1. Storage Layer — Embedded Qdrant

### Config additions

```python
# config.py
QDRANT_MODE: Literal["embedded", "http"] = "embedded"
QDRANT_PATH: str = "./storage/qdrant"
# QDRANT_URL retained for http mode fallback
```

### Factory function

```python
# storage/vector_store.py
def build_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.QDRANT_MODE == "embedded":
        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=settings.QDRANT_PATH)
    return QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
```

All existing call sites that construct `QdrantClient(settings.QDRANT_URL, ...)` directly (CLI commands, MCP server) must switch to this factory.

## 2. AppState — Shared State

### Dataclass

```python
# server/app_state.py
@dataclass
class AppState:
    settings: Settings
    qdrant_client: QdrantClient
    embedder: FlagEmbeddingEmbedder
    reranker: Reranker
    metadata_mgr: SourceMetadataManager
    vector_store: QdrantVectorStore
    query_engine: QueryEngine
    pipeline: IngestionPipeline
    health: HealthMonitor
    job_manager: JobManager
    mcp: FastMCP | None = None  # Set after construction (chicken-egg)
```

### Construction (two-phase to solve chicken-egg)

```python
@classmethod
async def create(cls, settings: Settings) -> "AppState":
    # Phase 1: Build all components
    client = build_qdrant_client(settings)
    embedder = FlagEmbeddingEmbedder(settings)
    reranker = Reranker(settings)
    metadata_mgr = SourceMetadataManager(settings, client)
    vector_store = QdrantVectorStore(settings, client, metadata_mgr)
    query_engine = QueryEngine(settings, embedder, vector_store, reranker)
    pipeline = IngestionPipeline(
        settings, DocumentLoader(settings),
        SemanticChunker(settings), embedder,
        vector_store, metadata_mgr,
    )
    job_manager = JobManager(pipeline)
    health = HealthMonitor(settings, client)
    await health.start()

    # Phase 2: Build state, then create mcp with self-reference
    state = cls(
        settings=settings, qdrant_client=client,
        embedder=embedder, reranker=reranker,
        metadata_mgr=metadata_mgr, vector_store=vector_store,
        query_engine=query_engine, pipeline=pipeline,
        job_manager=job_manager, health=health, mcp=None,
    )
    state.mcp = create_mcp_app(state)
    return state
```

### JobManager

```python
class JobManager:
    def __init__(self, pipeline: IngestionPipeline):
        self._pipeline = pipeline
        self._jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()  # Serialize index tasks to avoid OOM

    async def submit(self, path: Path, filename: str, tags: list[str]) -> str:
        """Submit an ingestion job. Must be called from an async context."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        self._jobs[job_id] = {
            "job_id": job_id, "filename": filename, "status": "pending",
            "chunks": 0, "error": None, "created_at": now,
            "completed_at": None, "failed_files": [],
        }
        asyncio.create_task(self._run(job_id, path, tags))
        self._evict_expired()
        return job_id

    async def _run(self, job_id: str, path: Path, tags: list[str]):
        self._jobs[job_id]["status"] = "processing"
        async with self._lock:
            try:
                report = await self._pipeline.run(paths=[path], tags=tags)
                self._jobs[job_id].update(
                    status="done" if not report.failed else "failed",
                    chunks=report.succeeded,
                    error=report.failed_files[0] if report.failed_files else None,
                    completed_at=datetime.now(timezone.utc),
                    failed_files=report.failed_files,
                )
                # File is intentionally kept in DATA_DIR after indexing.
                # Supports re-indexing (kh index --force) and orphan cleanup.
            except Exception as e:
                self._jobs[job_id].update(
                    status="failed", error=str(e),
                    completed_at=datetime.now(timezone.utc),
                )

    def get(self, job_id: str) -> dict | None:
        self._evict_expired()
        return self._jobs.get(job_id)

    def has_running_job(self) -> bool:
        return self._lock.locked()

    async def wait_until_idle(self, timeout: float = 300.0):
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    pass
        except asyncio.TimeoutError:
            structlog.get_logger().warning("shutdown_timeout_force_exit")

    def _evict_expired(self):
        now = datetime.now(timezone.utc)
        expired = [
            jid for jid, j in self._jobs.items()
            if j.get("completed_at")
            and (now - j["completed_at"]).total_seconds() > 3600
        ]
        for jid in expired:
            del self._jobs[jid]
```

TTL: 3600s (1 hour), lazy eviction on submit()/get(). Hardcoded as module-level `_JOB_TTL_SECONDS = 3600` — add to Settings only if tuning is needed later.

## 3. HTTP Upload Server

### Endpoints

**POST /upload**
- Content-Type: `multipart/form-data`
- Fields: `file` (binary, required), `tags` (string, optional, comma-separated)
- Auth: `Authorization: Bearer <KH_MCP_AUTH_TOKEN>` (only when token is set)
- Response `200`: `{"job_id": "...", "status": "pending"}`
- Response `400`: `{"error": "Unsupported format: .bin"}`
- Response `401`: `{"error": "Unauthorized"}`
- Response `413`: `{"error": "File exceeds max size: 200MB"}`

**GET /upload/status/{job_id}**
- Response `200`: `{"job_id": "...", "filename": "...", "status": "done|processing|pending|failed", "chunks": 15, "error": null, "created_at": "...", "completed_at": "...", "failed_files": []}`
- Response `404`: `{"error": "Job not found"}`

### Implementation

```python
# server/upload_server.py
# Import SUPPORTED_SUFFIXES from loaders to stay in sync —
# upload validates with the same set the pipeline accepts.
from knowledge_hub.ingestion.loaders import SUPPORTED_SUFFIXES

def create_upload_app(state: AppState) -> Starlette:
    app = Starlette(routes=[
        Route("/upload", _upload, methods=["POST"]),
        Route("/upload/status/{job_id}", _status, methods=["GET"]),
    ])
    app.state.kh = state
    return app
```

Handlers are module-level functions (not closures) accessing state via `request.app.state.kh` for testability.

### Filename safety

```python
def _safe_filename(filename: str) -> str:
    name = PurePosixPath(filename).name     # strip directory traversal
    return re.sub(r'[^\w\-.]', '_', name)   # replace special chars
```

Dedup: if `dest` already exists, prepend `{job_id[:8]}_`.

### Format validation

Format checked before reading file content to avoid loading large unsupported files.

## 4. MCP Server — Refactored

### create_mcp_app() now receives AppState

```python
def create_mcp_app(state: AppState) -> FastMCP:
    # Auth setup — same logic as before, using state.settings
    if state.settings.MCP_AUTH_TOKEN:
        from fastmcp.server.auth import StaticTokenVerifier
        verifier = StaticTokenVerifier(
            {state.settings.MCP_AUTH_TOKEN: {"client_id": "knowledge-hub", "scopes": []}}
        )
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if state.settings.MCP_HOST != "127.0.0.1":
            raise ValueError(
                "MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set. "
                "Set KH_MCP_AUTH_TOKEN to enable LAN access."
            )
        mcp = FastMCP("knowledge-hub")

    # IP allowlist middleware — same as existing mcp_server.py
    if state.settings.MCP_ALLOWED_IPS:
        ...  # identical to current implementation

    # Register tools — pass components from state (signature expanded)
    tools = create_tools(
        state.settings, state.query_engine, state.health,
        state.metadata_mgr, state.vector_store,
    )
    for name, fn in tools.items():
        mcp.add_tool(fn)

    return mcp
```

`run_mcp_server()` is removed — serve command uses `mcp.http_app()` + uvicorn instead.

### MCP Tools — expanded to 3

```python
# server/tools.py
def create_tools(settings, query_engine, health, metadata_mgr, vector_store) -> dict:
    # query_knowledge_base (existing, unchanged)
    # list_kb_sources (new)
    # get_kb_status (new)
```

**list_kb_sources**: Returns `[{filename, chunk_count, source_hash}]` via new `metadata_mgr.list_source_details()`.

**get_kb_status**: Returns `{model_loaded, qdrant, gpu_available, gpu_memory_free_mb, collection, total_chunks (queried on demand), total_sources (queried on demand)}`. Chunk/source counts are queried directly, not cached via HealthMonitor — they are management info, not health signals.

### metadata_mgr addition

```python
# storage/metadata.py
async def list_source_details(self) -> list[dict]:
    """Return full payload for all sources."""
    points, next_offset = self._client.scroll(
        collection_name=self._collection, limit=100, with_payload=True,
    )
    results = [p.payload for p in points]
    while next_offset:
        points, next_offset = self._client.scroll(
            collection_name=self._collection, offset=next_offset, limit=100,
            with_payload=True,
        )
        results.extend(p.payload for p in points)
    return results
```

Existing `list_sources()` is preserved for backward compatibility.

## 5. Unified Startup — `kh serve`

```python
# cli/main.py
@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--upload-port", default=None, type=int)
@click.option("--no-upload", is_flag=True)
def serve(host, port, upload_port, no_upload):
    settings = _get_settings()
    if host: settings.MCP_HOST = host
    if port: settings.MCP_PORT = port
    if upload_port: settings.UPLOAD_PORT = upload_port

    async def _main():
        state = await AppState.create(settings)
        if no_upload:
            config = uvicorn.Config(
                state.mcp.http_app(transport="streamable-http",
                                   stateless_http=True, json_response=True),
                host=settings.SERVER_HOST, port=settings.MCP_PORT,
            )
            await uvicorn.Server(config).serve()
        else:
            await _run_servers(state, settings)

    anyio.run(_main)


async def _run_servers(state: AppState, settings: Settings):
    mcp_app = state.mcp.http_app(
        transport="streamable-http", stateless_http=True, json_response=True,
    )
    upload_app = create_upload_app(state)

    mcp_config = uvicorn.Config(mcp_app, host=settings.SERVER_HOST,
                                 port=settings.MCP_PORT, log_level="warning")
    upload_config = uvicorn.Config(upload_app, host=settings.SERVER_HOST,
                                    port=settings.UPLOAD_PORT, log_level="warning")

    async with anyio.create_task_group() as tg:
        tg.start_soon(uvicorn.Server(mcp_config).serve)
        tg.start_soon(uvicorn.Server(upload_config).serve)

    await _shutdown(state)
```

Key: use `mcp.http_app()` (not `mcp.run()`) to get the ASGI app — `mcp.run()` calls `anyio.run()` internally and conflicts with the existing event loop.

### Graceful shutdown

```python
async def _shutdown(state: AppState):
    if state.job_manager.has_running_job():
        logger.info("waiting_for_running_job")
        await state.job_manager.wait_until_idle(timeout=300.0)
    state.qdrant_client.close()
    logger.info("shutdown_complete")
```

## 6. Config — Full List

```python
# config.py (new/changed fields only)
QDRANT_MODE: Literal["embedded", "http"] = "embedded"  # NEW
QDRANT_PATH: str = "./storage/qdrant"                   # NEW
UPLOAD_PORT: int = 8766                                  # NEW
UPLOAD_ENABLED: bool = True                              # NEW

@property
def SERVER_HOST(self) -> str:
    """Shared bind address for MCP and upload servers."""
    return self.MCP_HOST

# Unchanged but relevant:
MCP_HOST: str = "127.0.0.1"        # Shared with upload server via SERVER_HOST
MCP_PORT: int = 8765
MCP_AUTH_TOKEN: str | None = None  # Shared auth for MCP + upload
MAX_FILE_SIZE_MB: int = 200        # Reused for upload size limit
DATA_DIR: str = "./data"           # Upload destination
```

Upload host is NOT a separate config — it reuses `MCP_HOST`. Upload and MCP share the same network visibility requirement; a single setting prevents configuration skew.

## 7. Error Handling Matrix

| Scenario | HTTP Status | Response |
|----------|-------------|----------|
| Missing/incorrect auth token | 401 | `{"error": "Unauthorized"}` |
| File exceeds MAX_FILE_SIZE_MB | 413 | `{"error": "File exceeds max size: N MB"}` |
| Unsupported file suffix | 400 | `{"error": "Unsupported format: .ext"}` |
| No file field in request | 400 | `{"error": "No file provided"}` |
| Job created, processing later fails | 200 | Job status `failed` with `error` field |
| Non-existent job_id | 404 | `{"error": "Job not found"}` |

Format validation happens before reading file content, minimizing wasted I/O on bad requests.

## 8. Testing Strategy

| Layer | Test type | What to verify |
|-------|-----------|----------------|
| `build_qdrant_client()` | Unit | Embedded creates dir + returns path-based client; http returns url-based |
| `JobManager` | Unit (mock pipeline) | Submit returns job_id; status transitions pending→processing→done; lock serialization; eviction |
| `upload_server` | Integration (TestClient) | 401 without token; 413 oversized; 400 bad format; 200 upload → job_id; 404 unknown job |
| `create_mcp_app(state)` | Unit | Tools registered; auth wired; middleware present |
| `create_tools()` | Unit (mock components) | 3 tools returned; list_sources/get_status return expected shape |
| `kh serve --no-upload` | Integration | Starts MCP-only, responds to query |
| `kh serve` (full) | Integration | Both ports respond; upload → status → query round-trip |

## 9. What Is NOT in Scope

- **External agent Skill** — deferred until server API is stable
- **Job persistence** — jobs in memory, lost on restart (acceptable for MVP)
- **Multi-file upload** — one file per POST; agent can call multiple times
- **Upload progress streaming** — polling GET /status is sufficient
- **Web UI** — HTTP API only
- **HTTPS** — handled by reverse proxy (nginx/caddy) if needed
