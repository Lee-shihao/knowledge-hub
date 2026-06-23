# Task 13 Report: Integration Tests

**Status:** DONE

**Date:** 2026-06-24

## Summary

Implemented end-to-end integration tests that exercise the full pipeline with real Qdrant and FlagEmbedding on CPU. All 7 tests verify the ingest→query→rerank flow against live infrastructure.

## Files Created

### `tests/test_integration.py`
- `qdrant_available()` — runtime check for Qdrant on localhost:6333
- Module-level `pytestmark` — skips entire module if Qdrant is unreachable
- `settings` fixture — isolated test config with temp directories
- `integration_setup` fixture — wires all real components (QdrantClient, FlagEmbeddingEmbedder, Reranker, etc.) with cleanup
- `rtdoc_path` fixture — creates a test RTOS markdown document

### Integration Tests

| Test | What it verifies |
|------|-----------------|
| `test_full_ingest_and_query` | Ingest markdown → query "priority inheritance" → verify relevant results with reranking |
| `test_ingest_skips_unchanged_file` | Second ingestion of same file should skip (hash match) |
| `test_ingest_reingests_changed_file` | Modified file should be re-ingested (hash mismatch) |
| `test_orphan_cleanup` | Deleting source file + running empty pipeline cleans orphans |
| `test_query_with_source_filter` | Source filter returns matching results; non-matching filter returns empty |
| `test_multiple_file_ingestion` | Two files ingested → both succeed |
| `test_query_empty_collection` | Querying empty collection returns empty results gracefully |

### Key Fixes Applied During Development

1. **FusionQuery for RRF**: `hybrid_search` uses `FusionQuery(fusion="rrf")` for Qdrant prefetch merging — plain `prefetch` list without `query=FusionQuery` caused API errors
2. **UUID format normalization**: Qdrant returns UUIDs without hyphens; vector store tests now normalize with `uuid.UUID(hex=...).hex` for consistent comparison
3. **FlagReranker tokenizer compat**: `prepare_for_model` removed in transformers 5.x — fixed by pinning `transformers>=4.40,<5.0` in pyproject.toml

### Test Results

| Suite | Passed | Total |
|-------|--------|-------|
| Unit tests | 126 | 126 |
| Integration tests | 7 | 7 |
| **Total** | **133** | **133** |

### Infrastructure Requirements

- **Qdrant**: `docker run -p 6333:6333 qdrant/qdrant`
- **FlagEmbedding models**: First run downloads ~2.2GB (BAAI/bge-m3 + BAAI/bge-reranker-v2-m3)
- **CPU mode**: All tests run on CPU (`EMBED_DEVICE="cpu"`)
- **Runtime**: ~80s for full integration suite (model loading dominates)

## Concerns

1. **No GPU integration test**: All tests use CPU. GPU-specific behavior (OOM degradation, batch size reduction) is only unit-tested with mocks.
2. **No concurrency test**: Multiple simultaneous ingest/query operations are not tested.
3. **Test data is synthetic**: Only short markdown documents. Real-world documents (PDFs, large files, Unicode) are not covered.
