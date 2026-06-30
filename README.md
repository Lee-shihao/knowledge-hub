# Knowledge Hub

**Local-first Vector RAG knowledge base with MCP + HTTP upload interface.**

[中文文档](README_CN.md)

Knowledge Hub lets you ingest documents (Markdown, PDF, plain text, HTML), embed them with BGE-M3 dense+sparse vectors, store in Qdrant, and query via hybrid search + cross-encoder reranking — all running locally with no cloud API calls. External agents can upload files via HTTP and query knowledge via MCP.

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
git clone https://github.com/Lee-shihao/knowledge-hub.git && cd knowledge-hub
uv sync

# Activate the virtual environment (optional)
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

```bash
# Pull the image
docker pull saxiburry/knowledge-hub:latest

# Run
docker run -d \
  --name knowledge-hub \
  --gpus all \
  -p 8765:8765 \
  -p 8766:8766 \
  -v kh_data:/app \
  -e KH_SERVER_HOST=0.0.0.0 \
  -e KH_DATA_DIR=/app/data \
  -e KH_STORAGE_DIR=/app/storage \
  -e KH_QDRANT_PATH=/app/storage/qdrant \
  -e HF_HOME=/app/models \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e KH_SERVER_AUTH_TOKEN=your-secret-token \
  saxiburry/knowledge-hub:latest
```

Or use Docker Compose:

```yaml
services:
  knowledge-hub:
    image: saxiburry/knowledge-hub:latest
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - kh_data:/app
    environment:
      - KH_SERVER_HOST=0.0.0.0
      - KH_DATA_DIR=/app/data
      - KH_STORAGE_DIR=/app/storage
      - KH_QDRANT_PATH=/app/storage/qdrant
      - HF_HOME=/app/models
      - HF_ENDPOINT=https://hf-mirror.com
      - KH_SERVER_AUTH_TOKEN=${KH_SERVER_AUTH_TOKEN}
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  kh_data:
```

```bash
export KH_SERVER_AUTH_TOKEN=your-secret-token
docker compose up -d
```

> `kh_data` named volume persists all data (documents, Qdrant index, model cache ~2.2GB) across container restarts.

### Build Strategy

| Trigger | Image Tags |
|---------|-----------|
| push tag `v*` | `0.1.0`, `latest` |

```bash
git tag v0.2.0
git push origin v0.2.0
```

## MCP Server Usage

The MCP server exposes 3 tools over JSON-RPC (streamable-http transport).

| Tool | Description |
|------|-------------|
| `query_knowledge_base` | Semantic search with hybrid dense+sparse + cross-encoder rerank |
| `list_kb_sources` | List all indexed sources with chunk count and content hash |
| `get_kb_status` | System health (model, Qdrant, GPU) + collection statistics |

### Remote Access (LAN)

```bash
export KH_SERVER_AUTH_TOKEN=test-token-123
kh serve --host 0.0.0.0
```

```bash
# Query
curl -s -X POST http://192.168.30.125:8765/mcp \
  -H "Authorization: Bearer test-token-123" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"your question","top_k":5}}}'
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

Upload files via HTTP for automatic ingestion.

```
POST /upload                    GET /upload/status/{job_id}
Content-Type: multipart/form    Response:
  file: <binary> (required)     {
  tags: "tag1,tag2" (optional)    "job_id": "e3ce9f20b6fc",
                                  "filename": "test-upload.md",
Response:                         "status": "done",
  {"job_id": "e3ce9f20b6fc",      "chunks": 1,
   "status": "pending"}           "error": null,
                                  ...}
Supported formats: .md .txt
  .pdf .html .htm .docx .rst
```

| Status | Meaning |
|--------|---------|
| `pending` | Job queued |
| `processing` | Ingestion running (load → chunk → embed → store) |
| `done` | Successfully indexed, queryable immediately |
| `failed` | Error during ingestion (see `error` field) |

Upload and MCP share the same `KH_SERVER_AUTH_TOKEN`. On localhost (default) no auth is required.

## Agent Skill Integration

Install the Knowledge Hub skill for AI agents (Hermes, Claude Code, OpenClaw). The skill bundles a Python upload script — agents call it directly; no manual curl needed.

```bash
# Install the skill (one-liner)
curl -fsSL https://raw.githubusercontent.com/Lee-shihao/knowledge-hub/main/install_skill.sh | bash -s -- hermes

# Also supports: claude, openclaw
```

Installed structure:
```
~/.hermes/skills/knowledge-hub/
├── SKILL.md
└── scripts/
    └── upload.py
```

Then configure the required environment variables:

```bash
echo 'KNOWLEDGE_HUB_BASE_URL="http://<server-ip>:8766"' >> ~/.hermes/.env
echo 'KNOWLEDGE_HUB_TOKEN="your-token-here"' >> ~/.hermes/.env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `KNOWLEDGE_HUB_BASE_URL` | Yes | Upload server HTTP endpoint (port 8766) |
| `KNOWLEDGE_HUB_TOKEN` | Yes | Auth token matching `KH_SERVER_AUTH_TOKEN` on the server |

> These variables are for the agent skill, not the server. The token value must match `KH_SERVER_AUTH_TOKEN` configured on the server side.

## License

Apache-2.0
