# Knowledge Hub

**Local-first Vector RAG knowledge base with MCP + HTTP upload interface.**

[中文文档](README_CN.md)

Knowledge Hub lets you ingest documents (Markdown, PDF, plain text, HTML), embed them with BGE-M3 dense+sparse vectors, store in Qdrant, and query via hybrid search + cross-encoder reranking — all running locally with no cloud API calls. External agents can upload files via HTTP and query knowledge via MCP.

## Architecture

```
GPU Server — single process via kh serve
┌─────────────────────────────────────────────────────────┐
│  anyio task group                                       │
│                                                         │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ uvicorn :8765     │  │ uvicorn :8766              │  │
│  │ MCP Server        │  │ HTTP Upload Server         │  │
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
│  │  - reranker (BGE-reranker-v2-m3)                 │   │
│  │  - pipeline → IngestionPipeline                  │   │
│  │  - job_manager → async upload jobs               │   │
│  │  - query_engine → hybrid search + rerank         │   │
│  │  - qdrant_client → embedded Qdrant               │   │
│  │    (./storage/qdrant/ by default)                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

  External Agent (Claude Code / Herd / Hermes)
    → Upload files: HTTP POST :8766/upload
    → Query knowledge: MCP :8765
```

## Features

- **Hybrid search**: Dense vectors (BGE-M3) + sparse vectors (lexical weights) fused via Reciprocal Rank Fusion
- **Cross-encoder reranking**: BGE-reranker-v2-m3 re-scores top candidates for precision
- **Incremental ingestion**: Content-hash-based skip for unchanged files, automatic re-ingest on modification
- **Orphan cleanup**: Detects and removes vectors for deleted source files
- **Embedded Qdrant**: No external database required — Qdrant runs in-process (default), with optional external mode
- **HTTP upload server**: `POST /upload` (multipart/form-data) with async job polling via `GET /upload/status/{id}`
- **MCP server**: 3 tools — `query_knowledge_base`, `list_kb_sources`, `get_kb_status` — via FastMCP (streamable-http) with optional auth + IP filtering
- **CLI**: Full control via `kh` command — index, query, status, config, serve
- **CPU/GPU auto-switch**: FlagEmbedding auto-detects CUDA; falls back to CPU gracefully
- **OOM resilience**: Batch size auto-reduces on CUDA OOM, reset via `kh config reset-batch-size`

## Quick Start

### Prerequisites

- Python 3.12

No external services required — Qdrant runs embedded by default.

### Install

```bash
# Clone and install
git clone https://github.com/Lee-shihao/knowledge-hub.git && cd knowledge-hub
uv sync

# Activate the virtual environment (optional, for direct kh usage)
source .venv/bin/activate

# Or use via uv run (no activation needed)
uv run kh --help

# First run downloads models (~2.2GB)
kh index --path ./data
```

### Usage

```bash
# ---- Ingestion ----
kh index --path ./my-docs
kh index --path ./my-docs --tags "python,ml"  # with tags
kh index --force                              # re-ingest everything

# ---- Query ----
kh query "how does priority inheritance work?"
kh query "scheduling algorithms" -k 10        # top 10 results

# ---- Management ----
kh status                                     # collection stats
kh cleanup-orphans                            # remove vectors for deleted files
kh config show                                # current settings
kh config reset-batch-size                    # reset embedding batch size

# ---- Server ----
kh serve                                      # MCP (:8765) + HTTP upload (:8766)
kh serve --no-upload                          # MCP only
kh serve --host 0.0.0.0 --port 8765 --upload-port 8766
```

### Upload Files via HTTP

```bash
# Upload a file (no auth needed on localhost)
curl -X POST http://127.0.0.1:8766/upload \
  -F "file=@my-doc.pdf" \
  -F "tags=research,ml"

# Response: {"job_id":"abc123def456","status":"pending"}

# Poll job status
curl http://127.0.0.1:8766/upload/status/abc123def456

# Response: {"job_id":"...","filename":"my-doc.pdf","status":"done","chunks":15,...}
```

### Environment Variables

All settings use `KH_` prefix and can be configured via:

1. **Environment variables** (recommended for deployment):
   ```bash
   export KH_EMBED_DEVICE=cuda          # Use GPU (auto-enables fp16)
   kh index --path ./data
   ```

2. **`.env` file** (recommended for development):
   ```bash
   # Create .env file in project root
   cat > .env << 'EOF'
   KH_EMBED_DEVICE=cpu               # Force CPU (disable GPU, use fp32)
   KH_CHUNK_MAX_TOKENS=512
   KH_HYBRID_CANDIDATE_K=30
   EOF

   kh config show  # Verify settings
   ```

3. **CLI overrides** (for one-off changes):
   ```bash
   kh serve --host 0.0.0.0 --port 8765 --upload-port 8766
   ```

> **Tip**: `KH_EMBED_DEVICE` controls where embedding/reranking models run:
> - `auto` — auto-detect CUDA, fallback to CPU (default)
> - `cuda` — force GPU, auto-enables fp16 for faster inference
> - `cpu` — force CPU, uses fp32 (slower but no GPU required)

| Variable | Default | Description |
|----------|---------|-------------|
| `KH_SERVER_HOST` | `127.0.0.1` | Bind address for MCP and upload servers |
| `KH_MCP_PORT` | `8765` | MCP server port |
| `KH_UPLOAD_PORT` | `8766` | HTTP upload server port |
| `KH_UPLOAD_ENABLED` | `true` | Enable HTTP upload server on `kh serve` |
| `KH_SERVER_AUTH_TOKEN` | — | Auth token for MCP and upload (required if binding to non-localhost) |
| `KH_SERVER_ALLOWED_IPS` | `[]` | IP allowlist for MCP server |
| `KH_EMBED_MODEL` | `BAAI/bge-m3` | Embedding model HuggingFace ID |
| `KH_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model HuggingFace ID |
| `KH_EMBED_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KH_QDRANT_MODE` | `embedded` | Qdrant mode: `embedded` (local storage) or `http` (external server) |
| `KH_QDRANT_PATH` | `./storage/qdrant` | Embedded Qdrant data directory |
| `KH_QDRANT_URL` | `http://localhost:6333` | External Qdrant endpoint (used when `QDRANT_MODE=http`) |
| `KH_QDRANT_COLLECTION` | `knowledge_hub` | Collection name |
| `KH_CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `KH_CHUNK_OVERLAP` | `0.1` | Overlap ratio between chunks |
| `KH_EMBED_BATCH_SIZE` | `16` | Embedding batch size |
| `KH_MAX_FILE_SIZE_MB` | `200` | Max upload file size |
| `KH_HYBRID_CANDIDATE_K` | `20` | Candidates fetched before reranking |
| `KH_FINAL_TOP_K` | `5` | Final results after reranking |
| `KH_DATA_DIR` | `./data` | Document source directory |
| `KH_STORAGE_DIR` | `./storage` | Metadata storage directory |

## Docker Deployment

For users who prefer not to set up a Python environment.

### Pull the Image

```bash
# Latest stable release
docker pull saxiburry/knowledge-hub:latest

# Or a specific version
docker pull saxiburry/knowledge-hub:0.1.0
```

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
services:
  knowledge-hub:
    image: saxiburry/knowledge-hub:latest
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
      - kh_models:/app/models
    environment:
      - KH_EMBED_DEVICE=cpu
      - KH_SERVER_HOST=0.0.0.0
      - KH_DATA_DIR=/app/data
      - KH_STORAGE_DIR=/app/storage
      - KH_QDRANT_PATH=/app/storage/qdrant
      - HF_HOME=/app/models
      - HF_ENDPOINT=https://hf-mirror.com
    restart: unless-stopped

volumes:
  kh_models:
```

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

> `kh_models` is a named volume for HuggingFace model cache (~2.2GB). Keeps models persisted across container restarts.

### Usage

```bash
# Upload a file
curl -X POST http://localhost:8766/upload -F "file=@my-doc.md"

# Query
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"your question","top_k":5}}}'
```

### GPU Support

```yaml
environment:
  - KH_EMBED_DEVICE=cuda
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Build Strategy

| Trigger | Image Tags |
|---------|-----------|
| push tag `v*` | `0.1.0`, `latest` |

Versioning is managed via git tags. Stable releases are published only when a tag is pushed:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## MCP Server Usage

The MCP server exposes 3 tools over JSON-RPC (streamable-http transport). All examples use `curl` — no special client needed.

### MCP Tools

| Tool | Description |
|------|-------------|
| `query_knowledge_base` | Semantic search with hybrid dense+sparse + cross-encoder rerank |
| `list_kb_sources` | List all indexed sources with chunk count and content hash |
| `get_kb_status` | System health (model, Qdrant, GPU) + collection statistics |

### Remote Access (LAN)

Bind to non-localhost with an auth token (shared by MCP and upload):

```bash
# Start server
export KH_SERVER_AUTH_TOKEN=test-token-123
kh serve --host 0.0.0.0
# [info] server_starting mcp=http://0.0.0.0:8765/mcp upload=http://0.0.0.0:8766/upload
```

**List available tools:**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Response includes 3 tools: query_knowledge_base, list_kb_sources, get_kb_status
```

**Query the knowledge base:**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"BCM2835 SPI interface count","top_k":3}}}'

# Response (3 results, 743ms query time):
#   Top hit (score 7.19): "BCM2835 provides 2 SPI interfaces: SPI0 (standard, 2 CS)
#     and SPI1 (auxiliary, 3 CS)" — source: test-upload.md, section: "SPI Interface Count"
#   Second (score 5.29): bcm2835-arm-peripherals.pdf page 152 — chip datasheet
#   Third (score 3.63): test-upload.md overview section
```

**List indexed sources:**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_kb_sources"}}'

# Response: {"sources":[{"filename":"bcm2835-arm-peripherals.pdf","chunk_count":2560,...}],"count":1}
```

**Check system status:**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_kb_status"}}'

# Response:
#   model_loaded: true        qdrant: true        gpu_available: true
#   gpu_memory_free_mb: 14393 collection: knowledge_hub
#   total_chunks: 2564        total_sources: 1
```

### Local Access (no auth)

```bash
kh serve
# Same endpoints on 127.0.0.1, no Authorization header needed
```

### Configure in AI Clients

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://<server-ip>:8765/mcp",
      "transport": "streamable-http",
      "headers": {"Authorization": "Bearer your-secret-token"}
    }
  }
}
```

## HTTP Upload Server

Upload files via HTTP for automatic ingestion. The server validates formats, saves to the data directory, and indexes asynchronously.

### End-to-End Flow (verified on GPU server)

**1. Upload a file:**

```bash
$ curl -s -X POST http://192.168.30.125:8766/upload \
    -H "Authorization: Bearer test-token-123" \
    -F "file=@test-upload.md" \
    -F "tags=spi,bcm2835,test"

{"job_id":"e3ce9f20b6fc","status":"pending"}
```

**2. Poll job status until complete (~1 second for small files):**

```bash
$ curl -s http://192.168.30.125:8766/upload/status/e3ce9f20b6fc \
    -H "Authorization: Bearer test-token-123"

{
  "job_id": "e3ce9f20b6fc",
  "filename": "test-upload.md",
  "status": "done",
  "chunks": 1,
  "error": null,
  "created_at": "2026-06-25T10:38:36.562323+00:00",
  "completed_at": "2026-06-25T10:38:37.563172+00:00",
  "failed_files": []
}
```

**3. Query the uploaded content via MCP:**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"BCM2835 SPI interface count","top_k":3}}}'

# Top result (score 7.19) is the newly uploaded file
```

### API Reference

```
POST /upload                    GET /upload/status/{job_id}
Content-Type: multipart/form    Response:
  file: <binary> (required)     {
  tags: "tag1,tag2" (optional)    "job_id": "e3ce9f20b6fc",
                                  "filename": "test-upload.md",
Response:                         "status": "done",
  {"job_id": "e3ce9f20b6fc",      "chunks": 1,
   "status": "pending"}           "error": null,
                                  "created_at": "2026-06-25T10:38:36",
Supported formats: .md .txt      "completed_at": "2026-06-25T10:38:37",
  .pdf .html .htm .docx .rst      "failed_files": []
                                }
```

| Status | Meaning |
|--------|---------|
| `pending` | Job queued |
| `processing` | Ingestion running (load → chunk → embed → store) |
| `done` | Successfully indexed, queryable immediately |
| `failed` | Error during ingestion (see `error` field) |

Upload and MCP share the same `KH_SERVER_AUTH_TOKEN`. On localhost (default) no auth is required.

## Project Structure

```
src/knowledge_hub/
├── config.py              # Settings (pydantic-settings, KH_ env prefix)
├── schemas.py             # ChunkMetadata, DocumentChunk, QueryInput, QueryResult
├── cli/
│   └── main.py            # Click CLI: index, query, status, cleanup-orphans, config, serve
├── ingestion/
│   ├── chunker.py         # SemanticChunker — heading-aware splitting
│   ├── embedder.py        # FlagEmbeddingEmbedder — BGE-M3 dense+sparse
│   ├── loaders.py         # DocumentLoader — .md/.txt/.pdf with hash computation
│   └── pipeline.py        # IngestionPipeline — load→chunk→embed→store
├── retrieval/
│   ├── query_engine.py    # QueryEngine — embed→hybrid search→rerank
│   └── reranker.py        # Reranker — FlagReranker with graceful degradation
├── server/
│   ├── app_state.py       # AppState — shared component injection for MCP + upload
│   ├── health.py          # HealthMonitor — Qdrant + GPU background probing
│   ├── job_manager.py     # JobManager — async upload job tracking with serialization
│   ├── mcp_server.py      # FastMCP app wiring with auth + IP filtering
│   ├── tools.py           # MCP tools: query_knowledge_base, list_kb_sources, get_kb_status
│   └── upload_server.py   # HTTP upload app — POST /upload, GET /upload/status/{id}
└── storage/
    ├── metadata.py        # SourceMetadataManager — hash tracking, orphan cleanup
    └── vector_store.py    # QdrantVectorStore — hybrid search, upsert, delete
```

## Testing

```bash
# Unit tests (no external services needed)
pytest -m "not integration"

# Integration tests (requires Qdrant on localhost:6333, or set KH_QDRANT_MODE=embedded)
pytest tests/test_integration.py -v -s

# All tests
pytest
```

| Suite | Count | Requires |
|-------|-------|----------|
| Unit | ~175 | Nothing (mocked) |
| Integration | ~7 | Qdrant + FlagEmbedding models (~2.2GB download) |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FlagEmbedding | 1.4.0 | BGE-M3 embedding + BGE-reranker-v2-m3 |
| transformers | ≥4.40, <5.0 | Tokenizer for FlagReranker (5.x removed `prepare_for_model`) |
| qdrant-client | ≥1.12.0 | Vector storage + hybrid search (embedded mode) |
| fastmcp | ≥2.3.0 | MCP server framework |
| llama-index | ≥0.12.0 | Document readers |
| click | ≥8.0 | CLI framework |
| starlette | * | HTTP upload server |
| uvicorn | * | ASGI server for MCP + upload |
| anyio | * | Task group for dual-server startup |
| structlog | ≥24.0 | Structured logging |
| pydantic | ≥2.0 | Schema validation |
| pydantic-settings | ≥2.0 | Environment-based config |

## License

Apache-2.0
