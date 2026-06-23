# Codebase Index
> 2026-06-23 · 56 files · ~80.7k tokens total
>
> **How to use:** Read this file first. Navigate to the exact file you need,
> then read only that file. Do not read entire directories.

## Source

**src/knowledge_hub/**
- `__init__.py`
- `config.py` — Settings
- `schemas.py` — ChunkMetadata, DocumentChunk, QueryInput, ChunkResult, QueryResult

**src/knowledge_hub/cli/**
- `__init__.py`

**src/knowledge_hub/ingestion/**
- `__init__.py`
- `chunker.py` — SemanticChunker — heading-aware document chunking for embedding. · SemanticChunker
- `embedder.py` — OOMError, FlagEmbeddingEmbedder
- `loaders.py` — DocumentLoader

**src/knowledge_hub/retrieval/**
- `__init__.py`

**src/knowledge_hub/server/**
- `__init__.py`

**src/knowledge_hub/storage/**
- `__init__.py`
- `metadata.py` — SourceMetadataManager
- `vector_store.py` — QdrantVectorStore

**tests/**
- `__init__.py`
- `conftest.py` — temp_storage_dir
- `test_chunker.py` — Tests for SemanticChunker — heading-aware document chunking. · make_doc, test_chunker_produces_chunks, test_chunker_short_document, test_chunker_heading_path_in_metadata, test_chunk_id_deterministic, test_chunk_id_differs_for_different_source, test_heading_chain_resets_correctly, test_overlap_keeps_last_paragraph +2
- `test_config.py` — test_settings_defaults, test_settings_from_env
- `test_embedder.py` — settings, embedder, test_embed_query_returns_dense_and_sparse, test_embed_texts_batch, test_batch_size_persistence, test_reset_batch_size, test_embed_query_mocked, test_embed_texts_splits_into_batches +2
- `test_loaders.py` — test_compute_hash, test_load_markdown_file, test_load_text_file, test_load_nonexistent_file, test_large_file_warning, test_file_too_large_rejected, test_unsupported_suffix_skipped, test_supported_suffixes
- `test_metadata.py` — settings, metadata_mgr, test_upsert_and_get_hash, test_get_hash_missing, test_list_sources, test_remove, test_orphan_cleanup
- `test_schemas.py` — test_chunk_metadata_creation, test_chunk_metadata_defaults, test_document_chunk_excludes_embeddings, test_query_input_defaults, test_query_result_structure
- `test_vector_store.py` — make_chunk_id, settings, vector_store, test_ensure_collection, test_upsert_and_count, test_upsert_idempotent, test_delete_by_source, test_hybrid_search_no_filter +2

## Config
- `.claude/settings.json`
- `.claude/settings.local.json`
- `graphify-out/2026-06-23/.graphify_analysis.json`
- `graphify-out/2026-06-23/.graphify_labels.json`
- `graphify-out/2026-06-23/cost.json`
- `graphify-out/2026-06-23/graph.json`
- `graphify-out/2026-06-23/manifest.json`
- `pyproject.toml`

## Docs
- `.superpowers/sdd/task-2-brief.md`
- `.superpowers/sdd/task-2-report.md`
- `.superpowers/sdd/task-3-brief.md`
- `.superpowers/sdd/task-3-report.md`
- `.superpowers/sdd/task-4-brief.md`
- `.superpowers/sdd/task-4-report.md`
- `.superpowers/sdd/task-5-brief.md`
- `.superpowers/sdd/task-5-report.md`
- `.superpowers/sdd/task-6-brief.md`
- `.superpowers/sdd/task-6-report.md`
- `.superpowers/sdd/task-7-brief.md`
- `.superpowers/sdd/task-7-report.md`
- `CLAUDE.md`
- `CODEBASE_INDEX.md`
- `graphify-out/2026-06-23/GRAPH_REPORT.md`

---
*Index: ~902 tokens · Full codebase: ~80.7k tokens · Saves ~99%*
