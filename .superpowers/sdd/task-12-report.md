# Task 12 Report: CLI

**Status:** DONE

**Date:** 2026-06-24

## Summary

Implemented the command-line interface for the Vector RAG Knowledge Hub using Click. Provides 6 commands covering ingestion, querying, management, configuration, and server startup.

## Files Created

### `src/knowledge_hub/cli/main.py`
- `cli()` — Click group, entry point registered as `kh` via pyproject.toml
- `index()` — Ingest documents with `--path`, `--force`, `--tags` options
- `query()` — Query the knowledge base with `-k/--top-k` option
- `status()` — Show collection stats (chunk count, source files)
- `cleanup_orphans()` — Remove vectors for deleted source files
- `config show` — Display effective settings with auth token masking
- `config reset-batch-size` — Reset OOM-degraded batch size to default
- `serve()` — Start MCP server with `--host` / `--port` overrides
- Helper functions: `_get_settings()`, `_build_pipeline()`, `_build_query_engine()`

### Design Decisions
1. **Click over Typer**: Stable API, no pydantic v2 compatibility issues
2. **Async bridge**: Commands use `asyncio.run()` to call async pipeline/engine methods
3. **Token masking**: `config show` masks any field containing "auth_token" (first 4 chars + `****`)
4. **Status truncation**: Source file list capped at 20 with "... and N more" suffix

### Tests Created

| File | Tests | Passed |
|------|-------|--------|
| `tests/test_cli.py` | 15 | 15 |

Test classes:
- `TestCliHelp` (5) — help output for all commands
- `TestConfigShow` (3) — settings display, token masking, non-secret passthrough
- `TestStatus` (2) — Qdrant reachable/unreachable
- `TestConfigResetBatchSize` (1) — batch size reset
- `TestCleanupOrphans` (1) — orphan cleanup invocation
- `TestIndex` (2) — ingestion with no path, with tags
- `TestQuery` (2) — results display, top_k pass-through
- `TestServe` (1) — host/port override

## Concerns

1. **Token masking edge cases**: Current heuristic masks any field with "auth_token" in its name. Fields like `CHUNK_MAX_TOKENS` are correctly excluded (no "auth" in name), but a hypothetical `AUTH_TOKEN_COUNT` would be incorrectly masked.
2. **Duplicate QdrantClient creation**: Each command creates its own `QdrantClient` instance. For a CLI this is acceptable (short-lived process), but a connection pool would be needed for long-running use.
