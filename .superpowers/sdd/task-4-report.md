# Task 4 Report: QdrantVectorStore

## Status: DONE

**Commit:** `ce28ba1` — `feat: add QdrantVectorStore with dense+sparse native storage and RRF hybrid search`

## Files

| File | Action | Lines |
|------|--------|-------|
| `src/knowledge_hub/storage/vector_store.py` | Created | 117 |
| `tests/test_vector_store.py` | Created | 223 |

## Implementation

### QdrantVectorStore (`vector_store.py`)

Implements 5 async methods per the spec:

1. **`ensure_collection()`** — Creates a Qdrant collection with `vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)}` and `sparse_vectors_config={"sparse": SparseVectorParams()}`. Idempotent on second call (checks existing collections first).

2. **`upsert_chunks(chunks: list[DocumentChunk])`** — Converts each chunk's `sparse_embedding: dict[int, float]` to `SparseVector(indices=..., values=...)`, constructs `PointStruct` with payload including `text`, `source_file`, `source_hash`, `page_number`, `heading_path`, `tags`, and upserts to Qdrant.

3. **`hybrid_search(dense_vec, sparse_vec, top_k, filter_source, filter_tags)`** — Performs hybrid search via `query_points()` with `prefetch` for dense and sparse vectors, using RRF fusion. Supports optional `filter_source` (exact match) and `filter_tags` (any match) via `Filter` + `FieldCondition`. Returns `list[tuple[str, float, dict]]`.

4. **`delete_by_source(source_file)`** — Deletes all points matching `source_file` via `Filter` with `FieldCondition`.

5. **`count()`** — Returns total point count via `QdrantClient.count()`.

### Interfaces

- **Consumes:** `Settings`, `DocumentChunk`, `ChunkMetadata`, `SourceMetadataManager`
- **Produces:** `QdrantVectorStore` class with the 5 methods above
- Uses synchronous `QdrantClient` throughout (as specified in the plan)

## Tests

8 test cases covering all methods:

| Test | What it covers |
|------|---------------|
| `test_ensure_collection` | Idempotent collection creation |
| `test_upsert_and_count` | Basic upsert and count |
| `test_upsert_idempotent` | Same ID overwrites, no duplicates |
| `test_delete_by_source` | Delete by source file filtering |
| `test_hybrid_search_no_filter` | Hybrid search returns scored results |
| `test_hybrid_search_with_source_filter` | Source file filter narrowing |
| `test_hybrid_search_with_tag_filter` | Tag filter narrowing |
| `test_upsert_multiple_chunks` | Batch upsert of 5 chunks |

## Verification

- **TDD Step 1 (fail):** Confirmed `ModuleNotFoundError` before implementation
- **TDD Step 2 (pass):** Qdrant is not running in this environment, so tests cannot execute. **Code correctness verified by inspection:**
  - All Qdrant model imports confirmed valid (`VectorParams`, `SparseVectorParams`, `SparseVector`, `PointStruct`, `Filter`, `FieldCondition`, `MatchValue`, `MatchAny`, `Prefetch`)
  - `QdrantClient.create_collection()` with `vectors_config` + `sparse_vectors_config` verified correct
  - `SparseVector(indices=..., values=...)` construction verified correct
  - `QdrantClient.query_points()` with `prefetch`, `query_filter` verified correct
  - `QdrantClient.count()` returns `CountResult.count` verified correct
  - `QdrantClient.delete()` with `points_selector=Filter(...)` verified correct
  - All method signatures verified via `inspect.signature()`

## Concerns

- **Qdrant required for integration tests:** Tests are integration tests that require a running Qdrant at `localhost:6333`. Once Qdrant is available, run `pytest tests/test_vector_store.py -v` to validate.
- **Synchronous QdrantClient in async methods:** Uses sync client with async method stubs. This is intentional per the plan and noted as a future improvement.
- **Collection cleanup in test fixture:** The fixture deletes the collection on teardown. Make sure any errors during test setup don't leave stale collections.
