# Knowledge Hub

**Local-first Vector RAG knowledge base with MCP interface.**

[中文文档](README_CN.md)

Knowledge Hub lets you ingest documents (Markdown, plain text), embed them with BGE-M3 dense+sparse vectors, store in Qdrant, and query via hybrid search + cross-encoder reranking — all running locally with no cloud API calls.

## Architecture

```
┌──────────┐    ┌──────────────────────────────────────────────┐
│  CLI/MCP │───▶│  QueryEngine                                │
│  Server  │    │  embed → hybrid search (dense+sparse) → rerank│
└──────────┘    └──────────────────────────────────────────────┘
      │                         │              │
      ▼                         ▼              ▼
┌──────────┐            ┌────────────┐  ┌─────────────┐
│ Ingestion│            │ Qdrant      │  │ FlagReranker │
│ Pipeline │            │ Vector Store│  │ (BGE-v2-m3)  │
└──────────┘            └────────────┘  └─────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ Load → Chunk → Embed → Store        │
│  .md/.txt   Semantic  BGE-M3  Qdrant │
│             Chunker   (dense+  +meta │
│                       sparse)  store │
└─────────────────────────────────────┘
```

## Features

- **Hybrid search**: Dense vectors (BGE-M3) + sparse vectors (lexical weights) fused via Reciprocal Rank Fusion
- **Cross-encoder reranking**: BGE-reranker-v2-m3 re-scores top candidates for precision
- **Incremental ingestion**: Content-hash-based skip for unchanged files, automatic re-ingest on modification
- **Orphan cleanup**: Detects and removes vectors for deleted source files
- **MCP server**: Expose `query_knowledge_base` tool via FastMCP (SSE transport) with optional auth + IP filtering
- **CLI**: Full control via `kh` command — index, query, status, config, serve
- **CPU/GPU auto-switch**: FlagEmbedding auto-detects CUDA; falls back to CPU gracefully
- **OOM resilience**: Batch size auto-reduces on CUDA OOM, reset via `kh config reset-batch-size`

## Quick Start

### Prerequisites

- Python 3.12+
- [Qdrant](https://qdrant.tech/) running on localhost:6333

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Install

```bash
# Clone and install
git clone <repo-url> && cd knowledge-hub
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
# Ingest documents
kh index --path ./my-docs
kh index --path ./my-docs --tags "python,ml"  # with tags
kh index --force                              # re-ingest everything

# Query
kh query "how does priority inheritance work?"
kh query "scheduling algorithms" -k 10        # top 10 results

# Status
kh status

# Cleanup deleted files
kh cleanup-orphans

# Configuration
kh config show
kh config reset-batch-size

# Start MCP server
kh serve
kh serve --host 0.0.0.0 --port 9999
```

### Environment Variables

All settings use `KH_` prefix and can be configured via:

1. **Environment variables** (recommended for deployment):
   ```bash
   export KH_EMBED_DEVICE=cuda
   export KH_QDRANT_URL=http://qdrant-server:6333
   kh index --path ./data
   ```

2. **`.env` file** (recommended for development):
   ```bash
   # Create .env file in project root
   cat > .env << 'EOF'
   KH_EMBED_DEVICE=cuda
   KH_QDRANT_URL=http://localhost:6333
   KH_CHUNK_MAX_TOKENS=512
   KH_HYBRID_CANDIDATE_K=30
   EOF

   kh config show  # Verify settings
   ```

3. **CLI overrides** (for one-off changes):
   ```bash
   kh serve --host 0.0.0.0 --port 9999
   ```

| Variable | Default | Description |
|----------|---------|-------------|
| `KH_MCP_HOST` | `127.0.0.1` | MCP server bind address |
| `KH_MCP_PORT` | `8765` | MCP server port |
| `KH_MCP_AUTH_TOKEN` | — | Auth token (required if binding to non-localhost) |
| `KH_MCP_ALLOWED_IPS` | `[]` | IP allowlist for MCP server |
| `KH_EMBED_MODEL` | `BAAI/bge-m3` | Embedding model HuggingFace ID |
| `KH_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model HuggingFace ID |
| `KH_EMBED_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KH_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `KH_QDRANT_COLLECTION` | `knowledge_hub` | Collection name |
| `KH_CHUNK_MAX_TOKENS` | `512` | Max tokens per chunk |
| `KH_CHUNK_OVERLAP` | `0.1` | Overlap ratio between chunks |
| `KH_EMBED_BATCH_SIZE` | `16` | Embedding batch size |
| `KH_HYBRID_CANDIDATE_K` | `20` | Candidates fetched before reranking |
| `KH_FINAL_TOP_K` | `5` | Final results after reranking |
| `KH_DATA_DIR` | `./data` | Document source directory |
| `KH_STORAGE_DIR` | `./storage` | Metadata storage directory |

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
│   ├── loaders.py         # DocumentLoader — .md/.txt with hash computation
│   └── pipeline.py        # IngestionPipeline — load→chunk→embed→store
├── retrieval/
│   ├── query_engine.py    # QueryEngine — embed→hybrid search→rerank
│   └── reranker.py        # Reranker — FlagReranker with graceful degradation
├── server/
│   ├── health.py          # HealthMonitor — Qdrant + GPU background probing
│   ├── mcp_server.py      # FastMCP app wiring with auth + IP filtering
│   └── tools.py           # MCP tool: query_knowledge_base
└── storage/
    ├── metadata.py        # SourceMetadataManager — hash tracking, orphan cleanup
    └── vector_store.py    # QdrantVectorStore — hybrid search, upsert, delete
```

## Testing

```bash
# Unit tests (no external services needed)
pytest -m "not integration"

# Integration tests (requires Qdrant on localhost:6333)
pytest tests/test_integration.py -v -s

# All tests
pytest
```

| Suite | Count | Requires |
|-------|-------|----------|
| Unit | 126 | Nothing (mocked) |
| Integration | 7 | Qdrant + FlagEmbedding models (~2.2GB download) |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FlagEmbedding | 1.4.0 | BGE-M3 embedding + BGE-reranker-v2-m3 |
| transformers | ≥4.40, <5.0 | Tokenizer for FlagReranker (5.x removed `prepare_for_model`) |
| qdrant-client | ≥1.12.0 | Vector storage + hybrid search |
| fastmcp | ≥2.3.0 | MCP server framework |
| llama-index | ≥0.12.0 | Document readers |
| click | ≥8.0 | CLI framework |
| structlog | ≥24.0 | Structured logging |
| pydantic | ≥2.0 | Schema validation |
| pydantic-settings | ≥2.0 | Environment-based config |

## License

Apache-2.0
