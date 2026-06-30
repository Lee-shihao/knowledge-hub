# Codebase Index
> 2026-06-30 · 125 files · ~0.4M tokens total
>
> **How to use:** Read this file first. Navigate to the exact file you need,
> then read only that file. Do not read entire directories.

## Source

**skill/scripts/**
- `upload.py` — build_multipart, upload_file, poll_status, main

**src/knowledge_hub/**
- `__init__.py`
- `config.py` — Settings
- `schemas.py` — ChunkMetadata, DocumentChunk, QueryInput, ChunkResult, QueryResult

**src/knowledge_hub/cli/**
- `__init__.py`
- `main.py` — CLI for knowledge-hub: index, query, manage, and serve. · cli, index, query, status, cleanup_orphans, config, config_show, config_reset_batch_size +1

**src/knowledge_hub/ingestion/**
- `__init__.py`
- `chunker.py` — SemanticChunker — heading-aware document chunking for embedding. · SemanticChunker
- `embedder.py` — OOMError, FlagEmbeddingEmbedder
- `loaders.py` — DocumentLoader
- `pipeline.py` — IngestionPipeline — orchestrates the full ingestion flow: load → chunk → embed → store. · IngestionReport, IngestionPipeline

**src/knowledge_hub/retrieval/**
- `__init__.py`
- `query_engine.py` — QueryEngine — orchestrates the full query flow: embed → hybrid search → rerank. · QueryEngine
- `reranker.py` — Reranker using FlagEmbedding's FlagReranker for cross-encoder re-ranking. · Reranker

**src/knowledge_hub/server/**
- `__init__.py`
- `app_state.py` — AppState — single-process shared state for MCP and HTTP servers. · AppState
- `health.py` — HealthMonitor — background health prober for Qdrant and GPU. · HealthStatus, HealthMonitor
- `job_manager.py` — JobManager — async job tracking for ingestion pipeline execution. · JobManager
- `mcp_server.py` — MCP server — receives AppState and exposes query/list/status tools. · create_mcp_app
- `tools.py` — MCP tool wrappers around the QueryEngine. · create_tools
- `upload_server.py` — HTTP upload server — receives file uploads and queues ingestion jobs. · create_upload_app

**src/knowledge_hub/storage/**
- `__init__.py`
- `metadata.py` — SourceMetadataManager
- `vector_store.py` — build_qdrant_client, QdrantVectorStore

**tests/**
- `__init__.py`
- `conftest.py` — temp_storage_dir
- `test_chunker.py` — Tests for SemanticChunker — heading-aware document chunking. · make_doc, test_chunker_produces_chunks, test_chunker_short_document, test_chunker_heading_path_in_metadata, test_chunk_id_deterministic, test_chunk_id_differs_for_different_source, test_heading_chain_resets_correctly, test_overlap_keeps_last_paragraph +2
- `test_cli.py` — Tests for the knowledge-hub CLI. · runner, TestCliHelp, TestConfigShow, TestStatus, TestConfigResetBatchSize, TestCleanupOrphans, TestIndex, TestQuery +1
- `test_config.py` — test_settings_defaults, test_settings_from_env
- `test_embedder.py` — settings, embedder, test_embed_query_returns_dense_and_sparse, test_embed_texts_batch, test_batch_size_persistence, test_reset_batch_size, test_embed_query_mocked, test_embed_texts_splits_into_batches +2
- `test_health.py` — Tests for HealthMonitor (no real Qdrant/GPU probing, mock-based only). · TestHealthStatus, TestHealthMonitor
- `test_integration_upload.py` — Integration test — upload, status polling, query round-trip. · TestUploadQueryRoundTrip, TestUploadServerNoAuth
- `test_integration.py` — qdrant_available, settings, integration_setup, rtdoc_path, test_full_ingest_and_query, test_ingest_skips_unchanged_file, test_ingest_reingests_changed_file, test_orphan_cleanup +2
- `test_job_manager.py` — Tests for JobManager — async job tracking with serialized pipeline execution. · pipeline, job_manager, MockPipeline, TestJobManagerSubmit, TestJobManagerProcessing, TestJobManagerGet, TestJobManagerEviction, TestJobManagerShutdown
- `test_loaders.py` — test_compute_hash, test_load_markdown_file, test_load_text_file, test_load_nonexistent_file, test_large_file_warning, test_file_too_large_rejected, test_unsupported_suffix_skipped, test_supported_suffixes
- `test_mcp_server.py` — Tests for MCP server creation — app wiring, auth, IP middleware. · TestCreateMCPApp, TestStreamableHTTPIntegration
- `test_metadata.py` — settings, metadata_mgr, test_upsert_and_get_hash, test_get_hash_missing, test_list_sources, test_remove, test_orphan_cleanup, TestListSourceDetails
- `test_pipeline.py` — Tests for IngestionPipeline — orchestrates load → chunk → embed → store. · settings, mock_embedder, pipeline, test_pipeline_ingests_markdown, test_pipeline_skips_unchanged_file, test_pipeline_force_reingests, test_pipeline_handles_missing_file, test_pipeline_handles_unsupported_format +2
- `test_query_engine.py` — Tests for QueryEngine — orchestrates embed → hybrid search → rerank. · settings, embedder, vector_store, reranker, query_engine, test_query_empty_collection, test_query_with_results, test_query_passes_filters_to_hybrid_search +2
- `test_reranker.py` — Tests for Reranker using FlagReranker from FlagEmbedding. · settings, reranker, test_rerank_returns_top_k_sorted, test_rerank_preserves_metadata, test_rerank_empty_candidates, test_rerank_fewer_than_top_k, test_rerank_graceful_degradation_on_exception, test_rerank_uses_asyncio_to_thread +2
- `test_schemas.py` — test_chunk_metadata_creation, test_chunk_metadata_defaults, test_document_chunk_excludes_embeddings, test_query_input_defaults, test_query_result_structure
- `test_tools.py` — Tests for MCP tools — query_knowledge_base health gates and QueryEngine calls. · settings, mock_health, mock_query_engine, mock_metadata_mgr, mock_vector_store, TestCreateTools, TestQueryKnowledgeBase, TestListKbSources +1
- `test_upload_server.py` — Tests for HTTP upload server — upload, status, auth, validation. · settings, settings_with_auth, client, client_with_auth, TestUploadEndpoint, TestStatusEndpoint, TestFilenameSafety
- `test_vector_store.py` — make_chunk_id, settings, vector_store, test_ensure_collection, test_upsert_and_count, test_upsert_idempotent, test_delete_by_source, test_hybrid_search_no_filter +2

## Config
- `.claude/settings.json`
- `.claude/settings.local.json`
- `.github/workflows/docker-image.yml`
- `docker-compose.yml`
- `graphify-out/2026-06-23/.graphify_analysis.json`
- `graphify-out/2026-06-23/.graphify_labels.json`
- `graphify-out/2026-06-23/cost.json`
- `graphify-out/2026-06-23/graph.json`
- `graphify-out/2026-06-23/manifest.json`
- `graphify-out/2026-06-24/.graphify_analysis.json`
- `graphify-out/2026-06-24/.graphify_labels.json`
- `graphify-out/2026-06-24/cost.json`
- `graphify-out/2026-06-24/graph.json`
- `graphify-out/2026-06-24/manifest.json`
- `graphify-out/2026-06-25/.graphify_analysis.json`
- `graphify-out/2026-06-25/.graphify_labels.json`
- `graphify-out/2026-06-25/cost.json`
- `graphify-out/2026-06-25/graph.json`
- `graphify-out/2026-06-25/manifest.json`
- `pyproject.toml`

## Docs
- `.superpowers/sdd/task-7-report.md`
- `.superpowers/sdd/task-8-brief.md`
- `.superpowers/sdd/task-8-report.md`
- `.superpowers/sdd/task-9-brief.md`
- `.superpowers/sdd/task-9-report.md`
- `CLAUDE.md`
- `CODEBASE_INDEX.md`
- `data/algorithm-design-patterns.md`
- `data/c-best-practices.md`
- `graphify-out/2026-06-23/GRAPH_REPORT.md`
- `graphify-out/2026-06-24/GRAPH_REPORT.md`
- `graphify-out/2026-06-25/GRAPH_REPORT.md`
- `README_CN.md`
- `README.md`
- `skill/SKILL.md`

---
*Index: ~2.1k tokens · Full codebase: ~0.4M tokens · Saves ~100%*
