# Vector RAG Knowledge Hub — Design Spec

Date: 2026-06-23 | Status: Design Approved

## 1. Overview

Build a local-first Vector RAG knowledge hub for technical documents (datasheets, protocol specs, technical manuals, books). The system ingests mixed-format documents, indexes them into a vector database, and exposes retrieval via an MCP Server so Claude Code and other MCP-compatible agents can query the knowledge base autonomously.

**Target environment:** LAN-deployed, single GPU machine, multiple agent clients.

## 2. Technology Decisions

| Layer | Choice | Rationale |
|-------|--------|-----------|
| RAG Framework | **LlamaIndex** | Native document pipeline, better indexing abstractions than LangChain for this use case |
| Embedding Model | **bge-m3** via Ollama | Dense + sparse dual output, multilingual, 1024d, local GPU |
| Vector Store | **Qdrant** (local) | Native sparse vector support, hybrid search API, Rust performance |
| Reranker | **bge-reranker** via Ollama | Cross-encoder, local GPU |
| MCP Framework | **FastMCP 2.x** (pure Python) | Bearer auth built-in, Streamable HTTP ready, async-native |
| MCP Transport | **SSE** (primary), **Streamable HTTP** (reserved) | SSE for now; switchable via config |
| Config | **pydantic-settings** | .env / env var driven, typed |
| Data Models | **pydantic** | Shared schemas across CL CLI / MCP Server |
| Build System | **uv + pyproject.toml** | Modern Python tooling |
| Testing | **pytest + testcontainers** | Real Qdrant/Ollama containers in integration tests |

## 3. Project Structure

```
knowledge-hub/
├── docs/superpowers/specs/          # Design documents
├── src/knowledge_hub/
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings, all env vars
│   ├── schemas.py                   # DocumentChunk, ChunkMetadata, QueryInput, QueryResult
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # IngestionPipeline orchestrator
│   │   ├── loaders.py               # Format-dispatching document loaders
│   │   ├── chunker.py               # Semantic chunking + heading chain
│   │   └── embedder.py              # OllamaEmbedder: bge-m3 dense+sparse, OOM-aware
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # QdrantVectorStore: CRUD, sparse, orphan cleanup
│   │   └── metadata.py              # Source hash tracking for incremental updates
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_engine.py          # QueryEngine: orchestrates search -> rerank
│   │   └── reranker.py              # bge-reranker cross-encoder
│   ├── server/
│   │   ├── __init__.py
│   │   ├── mcp_server.py            # FastMCP server with auth
│   │   ├── tools.py                 # MCP Tool definitions (query_knowledge_base)
│   │   └── health.py                # Runtime health probes (Ollama, Qdrant, GPU)
│   └── cli/
│       ├── __init__.py
│       └── main.py                  # CLI: index, query, status, delete, config
├── tests/
│   ├── conftest.py                  # QdrantContainer, OllamaContainer fixtures
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   ├── test_ingestion_pipeline.py
│   ├── test_mcp_tools.py
│   └── test_integration.py          # Full ingestion + query e2e
├── data/                            # Source documents
├── storage/                         # Qdrant persistence + batch_size state
├── pyproject.toml
└── CLAUDE.md
```

## 4. Component Architecture

```
┌──────────────────────────────────────────────┐
│  MCP Server (FastMCP)  │  CLI (click/typer)  │  ← Presentation
│  - Bearer token auth   │  - index/query/     │
│  - Runtime health       │    status/delete    │
│  - Streamable HTTP ready│    config           │
├──────────────────────────────────────────────┤
│            QueryEngine (async)               │  ← Retrieval
│  embed query → hybrid search → rerank        │
├──────────────────────────────────────────────┤
│  IngestionPipeline  │  QdrantVectorStore     │  ← Ingestion + Storage
│  load→chunk→embed   │  CRUD + orphan cleanup │
├──────────────────────────────────────────────┤
│   LlamaIndex  │  Qdrant Client  │  Ollama    │  ← Infrastructure
└──────────────────────────────────────────────┘
```

Each component has a single purpose:
- **IngestionPipeline** — knows how to turn files into indexed vectors; knows nothing about querying
- **QueryEngine** — knows how to search and rerank; knows nothing about how documents got there
- **QdrantVectorStore** — owns all Qdrant I/O; the only place SparseVector conversion happens
- **MCP Server / CLI** — thin shells that wire components together

## 5. Data Models (schemas.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime

class ChunkMetadata(BaseModel):
    source_file: str
    source_hash: str                        # md5 of file content, for incremental updates
    page_number: int | None = None
    heading_path: list[str] = []            # ["Chapter 3", "3.2 Task Scheduling", "3.2.1 Priority Inheritance"]
    tags: list[str] = []
    ingested_at: datetime

class DocumentChunk(BaseModel):
    id: str                                 # md5(source_file + heading_path + text[:50]), deterministic
    text: str
    dense_embedding: list[float] = Field(exclude=True)   # 1024d, excluded from logs/serialization
    sparse_embedding: dict[int, float] = Field(exclude=True)  # converted to SparseVector in vector_store
    metadata: ChunkMetadata

class QueryInput(BaseModel):
    query: str
    top_k: int = 5
    filter_source: str | None = None
    filter_tags: list[str] | None = None

class ChunkResult(BaseModel):
    text: str
    source_file: str
    page_or_section: str
    heading_path: list[str]
    score: float

class QueryResult(BaseModel):
    results: list[ChunkResult]
    query_time_ms: float
```

## 6. Core Data Flows

### 6.1 Ingestion Pipeline

```
Raw files (data/*.pdf, *.md, *.html, *.docx, *.txt)
    │
    ▼
DocLoader ─── format detection → SimpleDirectoryReader dispatch
    │         PDF → markdown conversion preserves tables/code blocks
    │         Markdown/HTML/Word → parsed directly
    │
    ▼ List[Document]
Chunker ─── semantic split: prefer markdown headings / chapter markers
    │       max_tokens = 512 (hard limit, splits oversized tables/code blocks)
    │       overlay = 0.1 (10% overlap between chunks)
    │       preserves heading_path chain for context
    │
    ▼ List[DocumentChunk] (dense/sparse embeddings not yet set)
Embedder ─── Ollama / bge-m3 → {dense: 1024d float[], sparse: dict[int, float]}
    │       batch processing with OOM-aware auto-degradation
    │       max 3 retries with exponential backoff per batch
    │       single-text fallback if all batch attempts fail
    │
    ▼ List[DocumentChunk] (embeddings populated)
QdrantVectorStore ─── dict[int,float] → SparseVector(indices=[...], values=[...])
    │                 upsert with metadata payload
    │                 orphan cleanup: delete vectors for source_files not in local filesystem
    │
    ▼ Done — emit ingestion report: {total, succeeded, failed, skipped, orphans_cleaned}
```

**Metadata enrichment (tag assignment):**
Tags are populated from three sources, merged in priority order:
1. **Sidecar `.meta.json` files** — placed alongside documents, e.g. `data/rtos-book/.meta.json` with `{"tags": ["rtos", "embedded", "c"]}`. Highest priority.
2. **Directory name inference** — parent directory name is auto-added as a tag (e.g., `data/datasheets/` → tag `datasheets`). Can be disabled via config.
3. **CLI override** — `kh index --tags "tag1,tag2"` applies tags to all documents in that ingestion run.

Tags are stored in `ChunkMetadata.tags` and persisted in Qdrant payload for server-side filtering.

**Incremental update logic:**
1. Compute `md5(file_content)` for each local file
2. Query Qdrant for existing source_file → hash mapping
3. Skip files with matching hash; re-ingest files with changed hash
4. After ingestion, remove Qdrant points whose source_file no longer exists locally (orphan cleanup)

### 6.2 Query Pipeline

```
User query: "How does FreeRTOS implement priority inheritance?"
    │
    ▼
Embedder ─── query text → {dense: 1024d, sparse: dict[int, float]}
    │
    ▼
Qdrant ─── hybrid search with RRF (Reciprocal Rank Fusion)
    │      dense + sparse combined server-side
    │      top_k = 20 candidates
    │      optional: filter by source_file / tags
    │
    ▼ top 20
Reranker ─── bge-reranker cross-encoder reranks all 20
    │        returns top_k = 5
    │
    ▼
QueryResult ─── [{text, source_file, page, heading_path, score}]
```

Why RRF over weighted fusion: Qdrant's native RRF requires no manual weight tuning and typically matches or exceeds weighted fusion quality. Decision is configurable if future Qdrant versions add weighted query support.

## 7. Configuration (config.py)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KH_", env_file=".env")

    # Network
    MCP_HOST: str = "127.0.0.1"           # "0.0.0.0" only when MCP_AUTH_TOKEN is set
    MCP_PORT: int = 8765
    MCP_TRANSPORT: Literal["sse", "streamable-http"] = "sse"

    # Auth (REQUIRED for LAN deployment)
    MCP_AUTH_TOKEN: str | None = None      # None → forces MCP_HOST="127.0.0.1"
    MCP_ALLOWED_IPS: list[str] = []

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "bge-m3"
    RERANK_MODEL: str = "bge-reranker"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_hub"

    # Ingestion
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP: float = 0.1
    EMBED_BATCH_SIZE: int = 16           # initial; OOM degrades and persists lower
    MAX_FILE_SIZE_MB: int = 200
    WARN_FILE_SIZE_MB: int = 50

    # Query
    HYBRID_CANDIDATE_K: int = 20
    FINAL_TOP_K: int = 5

    # Storage paths (relative to project root or absolute)
    DATA_DIR: str = "./data"
    STORAGE_DIR: str = "./storage"
```

## 8. GPU OOM Handling (embedder.py)

```python
import asyncio
import json
from pathlib import Path

class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._effective_batch = self._load_persisted_batch_size()

    async def embed_batch(self, texts: list[str]) -> list[dict]:
        for attempt in range(3):
            try:
                return await self._call_ollama(texts, self._effective_batch)
            except OOMError:
                self._effective_batch = max(4, self._effective_batch // 2)
                self._persist_batch_size()
                await asyncio.sleep(2 ** attempt)
                continue
        # All batch attempts failed → single-text serial fallback
        results = []
        for t in texts:
            results.append(await self._call_ollama([t], 1))
        return results

    def _persist_batch_size(self):
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        state_file.write_text(json.dumps({"batch_size": self._effective_batch}))

    def _load_persisted_batch_size(self) -> int:
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())["batch_size"]
        return self.settings.EMBED_BATCH_SIZE
```

**Recovery:** Degraded batch_size persists across restarts. To reset to default, run: `kh config reset-batch-size`. Auto-increase is NOT used — OOM usually means genuine VRAM constraint.

## 9. MCP Server Security

```python
# server/mcp_server.py
from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier

def create_mcp_server(settings: Settings, query_engine: QueryEngine) -> FastMCP:
    if settings.MCP_AUTH_TOKEN:
        verifier = StaticTokenVerifier(settings.MCP_AUTH_TOKEN)
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if settings.MCP_HOST != "127.0.0.1":
            raise ValueError("MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set")
        mcp = FastMCP("knowledge-hub")

    mcp.add_tool(query_knowledge_base, name="query_knowledge_base")
    return mcp
```

**LAN deployment checklist:**
1. Set `KH_MCP_AUTH_TOKEN` to a strong random value
2. Set `KH_MCP_HOST=0.0.0.0` (only allowed when token is set)
3. Prefer Tailscale/WireGuard tunnel over bare TCP exposure
4. If behind a reverse proxy, add IP allowlist at proxy level

## 10. Runtime Health Checks (server/health.py)

```python
import asyncio
from dataclasses import dataclass

@dataclass
class HealthStatus:
    ollama: bool
    qdrant: bool
    gpu_available: bool
    gpu_memory_free_mb: int

class HealthMonitor:
    def __init__(self, settings: Settings, qdrant_client, ollama_client):
        self.settings = settings
        self._qdrant = qdrant_client
        self._ollama = ollama_client
        self._cached_status: HealthStatus | None = None
        self._task: asyncio.Task | None = None

    async def start(self, interval_seconds: int = 30):
        self._task = asyncio.create_task(self._probe_loop(interval_seconds))

    async def get_status(self) -> HealthStatus:
        if self._cached_status is None:
            self._cached_status = await self._probe_all()
        return self._cached_status

    async def _probe_loop(self, interval: int):
        while True:
            self._cached_status = await self._probe_all()
            await asyncio.sleep(interval)
```

QueryEngine checks `HealthMonitor.get_status()` before calling Ollama/Qdrant. If unhealthy, returns a clear error instead of timing out.

## 11. Error Handling Strategy

| Layer | Strategy |
|-------|----------|
| MCP Server | Catch all exceptions, return structured error JSON, never expose stack traces. Respect MCP error protocol. |
| Query Engine | Ollama timeout → retry (max 2). Qdrant unreachable → "Knowledge base unavailable." Empty collection → "Knowledge base is empty; ingest documents first." |
| Ingestion Pipeline | Single file failure does not stop the batch. Failed files collected in report. Embedding failure → exponential backoff retry (max 3), then log and skip. |
| Global | structlog for structured logging. Levels: DEBUG (dev), INFO (ops), WARNING (degradations), ERROR (failures). |

**Edge cases:**

| Scenario | Behavior |
|----------|----------|
| Ollama not running at startup | Fast fail with clear error message and start instructions |
| Qdrant empty on query | Return empty results with hint message |
| File > 50MB ingestion | Log warning, proceed |
| File > 200MB ingestion | Reject with error |
| Concurrent queries | Async MCP Server + Qdrant concurrent reads — handled natively |
| Duplicate file ingestion | source_hash match → skip |
| Orphan vectors (source deleted) | Post-ingestion cleanup pass |
| GPU OOM during embedding | Auto-degrade batch_size, persist, fallback to serial |

## 12. Testing Strategy

| Level | What | How |
|-------|------|-----|
| Unit | Chunker logic, embedder format, hash computation, schema validation | pytest |
| Integration | Qdrant CRUD, full ingestion→query loop, MCP tool invocation | pytest + testcontainers (QdrantContainer, OllamaContainer) |
| Contract | MCP Tool input/output schema conformance | pydantic validators |
| Manual | CLI commands, real Ollama models, MCP connection from Claude Code | Documented manual test steps |

### testcontainers fixtures (tests/conftest.py)

```python
import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer
from testcontainers.ollama import OllamaContainer

@pytest.fixture(scope="session")
def qdrant():
    with QdrantContainer("qdrant/qdrant:1.12.4") as qc:
        yield QdrantClient(
            host=qc.get_container_host_ip(),
            port=qc.get_exposed_port(6333)
        )

@pytest.fixture(scope="session")
def ollama():
    with OllamaContainer() as oc:
        # NOTE: first run pulls bge-m3 (~2GB). In CI, pre-cache the model layer.
        oc.start()
        yield oc.get_container_host_ip(), oc.get_exposed_port(11434)
```

## 13. MCP Tool Contract

```json
// Tool: query_knowledge_base
// Input schema:
{
  "query": "string (required) - natural language query",
  "top_k": "int (optional, default 5) - number of results to return",
  "filter_source": "string | null (optional) - filter by source filename",
  "filter_tags": "list[string] | null (optional) - filter by tags"
}

// Output schema:
{
  "results": [
    {
      "text": "string",
      "source_file": "string",
      "page_or_section": "string",
      "heading_path": ["string", ...],
      "score": "float"
    }
  ],
  "query_time_ms": "float"
}
```

Claude Code connects via MCP config:
```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://<host>:8765/sse",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

## 14. CLI Commands

```
kh index                # Ingest all documents from data/
kh index --path <dir>   # Ingest from specific directory
kh index --force        # Re-ingest all, ignoring source_hash cache
kh query "<question>"   # Direct query (no MCP needed)
kh query "<q>" -k 10    # With custom top_k
kh status               # Show document count, last ingestion time, storage size
kh cleanup-orphans       # Manually trigger orphan vector cleanup
kh config reset-batch-size  # Reset OOM-degraded batch size to default
kh config show           # Show current effective configuration
```

## 15. Out of Scope (for this spec)

- Multi-user authentication (single LAN trust domain)
- Document update via API (ingestion via CLI / file system only)
- Chat/conversational interface (single-shot retrieval only)
- Knowledge graph integration (graphify is separate)
- Document-level access control
- Monitoring dashboards / Prometheus metrics

---

*End of design spec.*
