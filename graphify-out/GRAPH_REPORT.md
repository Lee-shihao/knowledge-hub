# Graph Report - knowledge-hub  (2026-06-25)

## Corpus Check
- 82 files · ~61,017 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 968 nodes · 1388 edges · 84 communities (58 shown, 26 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 279 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e0d15672`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Project Documentation & Tooling|Project Documentation & Tooling]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]

## God Nodes (most connected - your core abstractions)
1. `Settings` - 108 edges
2. `FlagEmbeddingEmbedder` - 34 edges
3. `SourceMetadataManager` - 32 edges
4. `QdrantVectorStore` - 32 edges
5. `QueryInput` - 29 edges
6. `ChunkResult` - 27 edges
7. `QueryResult` - 24 edges
8. `SemanticChunker` - 23 edges
9. `HealthMonitor` - 22 edges
10. `DocumentChunk` - 20 edges

## Surprising Connections (you probably didn't know these)
- `pipeline()` --calls--> `IngestionPipeline`  [INFERRED]
  tests/test_pipeline.py → src/knowledge_hub/ingestion/pipeline.py
- `metadata_mgr()` --calls--> `SourceMetadataManager`  [INFERRED]
  tests/test_metadata.py → src/knowledge_hub/storage/metadata.py
- `settings()` --calls--> `Settings`  [EXTRACTED]
  tests/test_embedder.py → src/knowledge_hub/config.py
- `TestHealthMonitor` --uses--> `Settings`  [INFERRED]
  tests/test_health.py → src/knowledge_hub/config.py
- `TestHealthStatus` --uses--> `Settings`  [INFERRED]
  tests/test_health.py → src/knowledge_hub/config.py

## Import Cycles
- 1-file cycle: `src/knowledge_hub/server/mcp_server.py -> src/knowledge_hub/server/mcp_server.py`

## Communities (84 total, 26 thin omitted)

### Community 0 - "Project Documentation & Tooling"
Cohesion: 0.50
Nodes (4): Apache 2.0 License, CLAUDE.md Project Guidance File, graphify Knowledge Graph Tool, knowledge-hub Knowledge Management System

### Community 1 - "Community 1"
Cohesion: 0.33
Nodes (5): CLAUDE.md, Codebase Index, Config, Docs, Source

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (18): SemanticChunker — heading-aware document chunking for embedding., _build_page_or_section(), QueryEngine — orchestrates the full query flow: embed → hybrid search → rerank., Build page_or_section from candidate dict.      Uses heading_path[-1] if availab, Reranker using FlagEmbedding's FlagReranker for cross-encoder re-ranking., HealthStatus, HealthMonitor — background health prober for Qdrant and GPU.  Caches status and, Health status of system components. (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (19): 10. Runtime Health Checks (server/health.py), 11. Error Handling Strategy, 12. Testing Strategy, 13. MCP Tool Contract, 14. CLI Commands, 15. Out of Scope (for this spec), 1. Overview, 2. Technology Decisions (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (19): 10. 运行时健康检查（server/health.py）, 11. 错误处理策略, 12. 测试策略, 13. MCP Tool 契约, 14. CLI 命令, 15. 本次范围外（不在本规格中）, 1. 概述, 2. 技术选型 (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (16): Global Constraints, Task 10: Retrieval — QueryEngine, Task 11: Server — Health Monitor + MCP Tools + MCP Server, Task 12: CLI, Task 13: Integration Tests, Task 14: Final Verification & Graphify Update, Task 1: Project Scaffolding, Task 2: Config (config.py) + Schemas (schemas.py) (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (38): Split text into chunks respecting max_tokens, with overlap., Split an oversized text element into max_tokens-sized chunks by lines., Create a DocumentChunk with a deterministic ID., Split text by markdown headings, tracking the heading chain., ChunkMetadata, DocumentChunk, Document, DocumentChunk (+30 more)

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (5): Commits, Notes, Status: DONE, Task 2 Report: build_qdrant_client() factory function, Test Result

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Commits, Concerns, Status, Test Summary

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (4): Codebase Navigation, Current State, graphify, Project Overview

### Community 10 - "Community 10"
Cohesion: 0.25
Nodes (7): Architecture Change: Ollama → FlagEmbedding, CLI Smoke Test, Resolved Issues, SDD Progress Ledger, Task 14: Final Verification, Tasks, Test Results

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (11): FlagEmbeddingEmbedder, OOMError, Embed a single query text. Returns {dense, sparse}., Embed a batch of texts. Each result is {dense, sparse}., Encode texts using BGEM3FlagModel, returning dense + sparse vectors., Raised when the GPU runs out of memory during embedding., Convert FlagEmbedding lexical_weights {str: float} to Qdrant-compatible {int: fl, Wraps FlagEmbedding's BGEM3FlagModel for dense + sparse embedding generation. (+3 more)

### Community 25 - "Community 25"
Cohesion: 0.05
Nodes (53): BaseSettings, Shared bind address for MCP and upload servers., Settings, Initialize the health monitor.          Args:             settings: Application, QdrantClient, Settings, QdrantClient, Settings (+45 more)

### Community 26 - "Community 26"
Cohesion: 0.05
Nodes (33): embedder(), embed_texts splits texts into chunks of _effective_batch., _encode_batch returns dense + sparse for each input text., _encode_batch should raise OOMError when GPU runs out of memory., On repeated OOM, falls back to serial single-text calls., After one OOM, batch size is halved and retry succeeds., _convert_sparse converts lexical_weights {str: float} to {int: float} via MD5., _convert_sparse on empty dict returns empty dict. (+25 more)

### Community 27 - "Community 27"
Cohesion: 0.07
Nodes (37): mock_embedder(), pipeline(), Tests for IngestionPipeline — orchestrates load → chunk → embed → store., Missing files should not be counted as failures (loader skips them)., Unsupported file formats should be skipped, not counted as failures., Sidecar .meta.json should provide tags with highest priority., Parent directory name should be used as fallback tag., CLI tags should be applied when no sidecar exists. (+29 more)

### Community 28 - "Community 28"
Cohesion: 0.21
Nodes (9): FastMCP, create_mcp_app(), MCP server — wires together all components and exposes query_knowledge_base tool, Start the MCP server with health monitoring.      This is a synchronous function, Build and configure the FastMCP application.      Wires together all components:, run_mcp_server(), Settings, streamable-http transport should pass stateless_http=True and json_response=True (+1 more)

### Community 29 - "Community 29"
Cohesion: 0.06
Nodes (29): embedder(), query_engine(), Tests for QueryEngine — orchestrates embed → hybrid search → rerank., Query passes filter_source and filter_tags to hybrid_search., Settings with test defaults., Hybrid search uses HYBRID_CANDIDATE_K; rerank uses top_k from QueryInput., When reranker fails, results still returned (graceful degradation)., Mock FlagEmbeddingEmbedder. (+21 more)

### Community 30 - "Community 30"
Cohesion: 0.07
Nodes (27): Tests for Reranker using FlagReranker from FlagEmbedding., When compute_score raises an exception, reranker returns candidates unchanged., Settings with CPU device for tests., rerank wraps synchronous compute_score in asyncio.to_thread., _resolve_device returns 'cuda' when available., _resolve_device returns 'cpu' when CUDA is not available., _resolve_device returns explicit device when not 'auto'., Reranker uses fp16=True on CUDA device. (+19 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (24): _build_pipeline(), _build_query_engine(), cleanup_orphans(), cli(), config(), config_reset_batch_size(), config_show(), _get_settings() (+16 more)

### Community 32 - "Community 32"
Cohesion: 0.09
Nodes (22): integration_setup(), qdrant_available(), End-to-end integration test: ingest a markdown file and query it.  Requires: - Q, Create a test RTOS document., Ingest a markdown document and verify it can be queried with real embeddings., Re-ingesting the same file should skip it., Changed file should be re-ingested., Orphan cleanup should remove vectors for deleted files. (+14 more)

### Community 33 - "Community 33"
Cohesion: 0.09
Nodes (12): Exception, Tests for `kh status`., TestStatus, Tests for HealthMonitor (mocked)., Mock QdrantClient for testing., _probe_qdrant should return True when Qdrant responds., _probe_qdrant should return False on exception., _probe_gpu should return (bool, int) tuple. (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.09
Nodes (21): HealthMonitor, QueryEngine, HealthMonitor, Probe GPU availability and free memory via nvidia-smi.          Returns:, Background health prober for Qdrant and GPU.      No Ollama probing — FlagEmbedd, Start the background probe loop.          Args:             interval_seconds: In, Get the current cached health status.          If not yet cached, probes once an, Background loop that periodically probes all components.          Args: (+13 more)

### Community 35 - "Community 35"
Cohesion: 0.11
Nodes (9): Tests for the knowledge-hub CLI.  Uses Click's CliRunner for testing CLI command, Tests for `kh config reset-batch-size`., Tests for `kh cleanup-orphans`., Tests for `kh index`., Tests for `kh config show`., TestCleanupOrphans, TestConfigResetBatchSize, TestConfigShow (+1 more)

### Community 36 - "Community 36"
Cohesion: 0.22
Nodes (18): BaseModel, ChunkResult, QueryInput, QueryResult, QueryInput, QueryResult, Reranker, QueryEngine (+10 more)

### Community 37 - "Community 37"
Cohesion: 0.06
Nodes (34): Architecture, Knowledge Hub 知识库, MCP 服务器使用, 传输协议选项, 使用方法, 依赖, 前置条件, 功能特性 (+26 more)

### Community 38 - "Community 38"
Cohesion: 0.18
Nodes (10): API Adaptations for FastMCP 3.4.2, Critical Corrections Applied, Files Created, Pre-existing Test Failures (Not Caused by This Task), `src/knowledge_hub/server/health.py`, `src/knowledge_hub/server/mcp_server.py`, `src/knowledge_hub/server/tools.py`, Summary (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (10): 1. `src/knowledge_hub/retrieval/reranker.py`, 2. `tests/test_reranker.py`, Commit Hash, Implementation Summary, Issues Found and Fixed, No Issues Found, Self-Review, Status (+2 more)

### Community 40 - "Community 40"
Cohesion: 0.20
Nodes (9): Concerns, Files, Implementation, Interfaces, QdrantVectorStore (`vector_store.py`), Status: DONE, Task 4 Report: QdrantVectorStore, Tests (+1 more)

### Community 41 - "Community 41"
Cohesion: 0.25
Nodes (7): Concerns, Files Changed, FlagEmbedding Transition Report, Import Verification, Status: COMPLETE, Summary, Test Results

### Community 43 - "Community 43"
Cohesion: 0.33
Nodes (5): Commit Hash, Implementation Summary, Self-Review, Task 10 Report: Retrieval — QueryEngine, Test Results

### Community 44 - "Community 44"
Cohesion: 0.29
Nodes (4): Initialize the reranker with the configured model.          Args:             se, Ensure model is cached locally.          This prevents download failures when lo, Resolve 'auto' to 'cuda' if available, else 'cpu'.          Args:             em, Settings

### Community 46 - "Community 46"
Cohesion: 0.07
Nodes (29): Always Check malloc Return Value, Always NUL-Terminate Strings, Avoid Dangling Pointers, Avoid Repeated Calculations in Loops, Buffer Safety, C Programming Best Practices, Check for NULL Before Dereferencing, Check Return Values (+21 more)

### Community 60 - "Community 60"
Cohesion: 0.07
Nodes (27): 1. Storage Layer — Embedded Qdrant, 2. AppState — Shared State, 3. HTTP Upload Server, 4. MCP Server — Refactored, 5. Unified Startup — `kh serve`, 6. Config — Full List, 7. Error Handling Matrix, 8. Testing Strategy (+19 more)

### Community 61 - "Community 61"
Cohesion: 0.10
Nodes (19): Algorithm Design Patterns, Approaches, Binary Search, Binary Search Trees, Classic Problems, Data Structures, Divide and Conquer, Dynamic Programming (+11 more)

### Community 62 - "Community 62"
Cohesion: 0.29
Nodes (16): DocumentLoader, Splits documents into semantic chunks for embedding.      Strategy:     1. Split, SemanticChunker, DocumentLoader, Loads documents from files, dispatching by format.      PDFs are extracted with, IngestionPipeline, Orchestrates the full ingestion flow: load → chunk → embed → store.      Handles, SemanticChunker (+8 more)

### Community 63 - "Community 63"
Cohesion: 0.12
Nodes (15): 1. `config.py` — 更新默认传输, 2. `mcp_server.py` — 传递 stateless 和 json_response 参数, 3. README.md / README_CN.md — 更新使用示例, 4. 测试更新, Agent 集成配置, Claude Code (`.claude/settings.json`), curl 测试示例（切换后）, Hermes / OpenClaw / Codex (+7 more)

### Community 64 - "Community 64"
Cohesion: 0.13
Nodes (8): curl 风格的集成测试 — 验证 streamable-http 传输协议。      使用 Starlette TestClient 直接向 FastMCP, curl 风格 POST /mcp 调用 tools/list，应返回工具列表。, json_response=True 时应返回纯 JSON，而非 SSE 文本流。, MCP initialize 请求应可通过直接 POST 完成。, 无效的 JSON-RPC 方法应返回错误响应。, 缺少 Accept: application/json 时应返回 406 Not Acceptable。, curl 风格 tools/call query_knowledge_base 应返回查询结果。, TestStreamableHTTPIntegration

### Community 65 - "Community 65"
Cohesion: 0.14
Nodes (12): Global Constraints, HTTP Upload Server + Unified Architecture Implementation Plan, Task 10: Integration test — full upload → status → query round-trip, Task 11: Final verification — run full suite, Task 1: Config — Add new fields and SERVER_HOST property, Task 2: build_qdrant_client() factory function, Task 4: JobManager — async job tracking with serialized execution, Task 5: AppState — shared state dataclass (+4 more)

### Community 66 - "Community 66"
Cohesion: 0.18
Nodes (6): DocumentChunk, Remove vectors for files no longer on disk.          Returns 0 if the metadata c, Manages source file metadata in a separate Qdrant collection.      Uses a lightw, Return full payload for all sources.          Unlike list_sources() which return, SourceMetadataManager, Upsert document chunks with both dense and sparse embeddings.

### Community 67 - "Community 67"
Cohesion: 0.14
Nodes (5): metadata_mgr(), Existing list_sources() should still return just filenames., Tests for list_source_details() — returns full payload per source., settings(), TestListSourceDetails

### Community 68 - "Community 68"
Cohesion: 0.15
Nodes (12): `_estimate_tokens` → `_count_tokens`, `_hard_split` — 逐行 token 计数切分, `__init__`, 修改文件, 各语言 chars/token 差异, 向后兼容, 影响范围, 性能 (+4 more)

### Community 69 - "Community 69"
Cohesion: 0.17
Nodes (5): Tests for create_mcp_app() — app creation and configuration., create_mcp_app should return a FastMCP instance., When MCP_AUTH_TOKEN is set, auth should be configured without error., When MCP_HOST is not 127.0.0.1 and no auth token, should raise ValueError., TestCreateMCPApp

### Community 70 - "Community 70"
Cohesion: 0.20
Nodes (9): Concerns, Files Created, Infrastructure Requirements, Integration Tests, Key Fixes Applied During Development, Summary, Task 13 Report: Integration Tests, Test Results (+1 more)

### Community 71 - "Community 71"
Cohesion: 0.20
Nodes (6): _patch_heavy_components(), Return a context manager that patches all heavy server components.      Patches:, 创建 MCP app 并返回 Starlette TestClient（正确处理 lifespan）。, When MCP_HOST is 127.0.0.1 and no auth token, should succeed., When MCP_ALLOWED_IPS is set, IP middleware should be added without error., create_mcp_app should register query_knowledge_base tool.

### Community 72 - "Community 72"
Cohesion: 0.25
Nodes (7): Concerns, Design Decisions, Files Created, `src/knowledge_hub/cli/main.py`, Summary, Task 12 Report: CLI, Tests Created

### Community 73 - "Community 73"
Cohesion: 0.38
Nodes (4): Compute MD5 hash of file content for incremental update detection., Load a single PDF, one Document per page, with garbage detection., Document, Path

### Community 74 - "Community 74"
Cohesion: 0.29
Nodes (5): IngestionReport, IngestionPipeline — orchestrates the full ingestion flow: load → chunk → embed →, Summary of an ingestion run., Run the ingestion pipeline.          Args:             paths: List of file paths, Path

### Community 75 - "Community 75"
Cohesion: 0.29
Nodes (4): Tests for `kh serve`., Default MCP_TRANSPORT should be streamable-http, propagated to run_mcp_server., When MCP_TRANSPORT is set to sse, it should be propagated., TestServe

### Community 76 - "Community 76"
Cohesion: 0.29
Nodes (4): Tests for MCP server creation — app wiring, auth, IP middleware.  All heavy comp, Tests for IP allowlist middleware behavior., App with IP middleware should be created successfully., TestIPMiddleware

### Community 77 - "Community 77"
Cohesion: 0.33
Nodes (5): Global Constraints, Task 1: 更新 `__init__` — 加载 tokenizer 并创建 `_count_tokens`, Task 2: 替换 `_estimate_tokens` 调用 + 重写 `_hard_split`, Task 3: 更新测试, 精确 Token 计数 Implementation Plan

## Knowledge Gaps
- **247 isolated node(s):** `knowledge-hub`, `Status: COMPLETE`, `Summary`, `Files Changed`, `Test Results` (+242 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 25` to `Community 2`, `Community 6`, `Community 24`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`, `Community 32`, `Community 33`, `Community 34`, `Community 36`, `Community 44`, `Community 62`, `Community 64`, `Community 66`, `Community 67`, `Community 69`, `Community 73`, `Community 74`, `Community 76`, `Community 78`?**
  _High betweenness centrality (0.191) - this node is a cross-community bridge._
- **Why does `ChunkResult` connect `Community 36` to `Community 64`, `Community 33`, `Community 34`, `Community 35`, `Community 69`, `Community 6`, `Community 42`, `Community 75`, `Community 76`, `Community 45`, `Community 25`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `QueryResult` connect `Community 36` to `Community 64`, `Community 33`, `Community 34`, `Community 35`, `Community 69`, `Community 6`, `Community 42`, `Community 75`, `Community 76`, `Community 45`, `Community 25`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 58 inferred relationships involving `Settings` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`Settings` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `FlagEmbeddingEmbedder` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`FlagEmbeddingEmbedder` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `SourceMetadataManager` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`SourceMetadataManager` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `QdrantVectorStore` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`QdrantVectorStore` has 24 INFERRED edges - model-reasoned connections that need verification._