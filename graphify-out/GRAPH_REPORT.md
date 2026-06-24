# Graph Report - knowledge-hub  (2026-06-24)

## Corpus Check
- 71 files · ~37,446 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 733 nodes · 1136 edges · 60 communities (38 shown, 22 thin omitted)
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 270 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `13d56054`
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
- [[_COMMUNITY_Community 14|Community 14]]
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

## God Nodes (most connected - your core abstractions)
1. `Settings` - 96 edges
2. `FlagEmbeddingEmbedder` - 35 edges
3. `QdrantVectorStore` - 33 edges
4. `QueryInput` - 31 edges
5. `SourceMetadataManager` - 31 edges
6. `SemanticChunker` - 26 edges
7. `ChunkResult` - 22 edges
8. `QueryResult` - 20 edges
9. `HealthMonitor` - 20 edges
10. `DocumentLoader` - 19 edges

## Surprising Connections (you probably didn't know these)
- `metadata_mgr()` --calls--> `SourceMetadataManager`  [INFERRED]
  tests/test_metadata.py → src/knowledge_hub/storage/metadata.py
- `settings()` --calls--> `Settings`  [EXTRACTED]
  tests/test_embedder.py → src/knowledge_hub/config.py
- `TestHealthMonitor` --uses--> `Settings`  [INFERRED]
  tests/test_health.py → src/knowledge_hub/config.py
- `TestHealthStatus` --uses--> `Settings`  [INFERRED]
  tests/test_health.py → src/knowledge_hub/config.py
- `settings()` --calls--> `Settings`  [EXTRACTED]
  tests/test_integration.py → src/knowledge_hub/config.py

## Import Cycles
- 1-file cycle: `src/knowledge_hub/server/mcp_server.py -> src/knowledge_hub/server/mcp_server.py`

## Communities (60 total, 22 thin omitted)

### Community 0 - "Project Documentation & Tooling"
Cohesion: 0.50
Nodes (4): Apache 2.0 License, CLAUDE.md Project Guidance File, graphify Knowledge Graph Tool, knowledge-hub Knowledge Management System

### Community 1 - "Community 1"
Cohesion: 0.33
Nodes (5): CLAUDE.md, Codebase Index, Config, Docs, Source

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (20): SemanticChunker — heading-aware document chunking for embedding., _build_page_or_section(), QueryEngine — orchestrates the full query flow: embed → hybrid search → rerank., Build page_or_section from candidate dict.      Uses heading_path[-1] if availab, Reranker using FlagEmbedding's FlagReranker for cross-encoder re-ranking., HealthStatus, HealthMonitor — background health prober for Qdrant and GPU.  Caches status and, Health status of system components. (+12 more)

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
Cohesion: 0.09
Nodes (29): Split an oversized text element into max_tokens-sized chunks., Create a DocumentChunk with a deterministic ID., Rough token estimation: ~4 chars per token., Split text by markdown headings, tracking the heading chain., Split text into chunks respecting max_tokens, with overlap., ChunkMetadata, DocumentChunk, Document (+21 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (8): Commits, Concerns, Deviations from Brief, Files Created/Modified, Status: DONE, Task 2 Report: Config + Schemas, TDD Steps Followed, Test Summary

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (6): Commits, Concerns, Self-Review, Task 1 Report, Tests, What was implemented

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (4): Codebase Navigation, Current State, graphify, Project Overview

### Community 10 - "Community 10"
Cohesion: 0.25
Nodes (7): Architecture Change: Ollama → FlagEmbedding, CLI Smoke Test, Known Issues, SDD Progress Ledger, Task 14: Final Verification, Tasks, Test Results

### Community 24 - "Community 24"
Cohesion: 0.06
Nodes (45): DocumentLoader, Splits documents into semantic chunks for embedding.      Strategy:     1. Split, SemanticChunker, FlagEmbeddingEmbedder, OOMError, Convert FlagEmbedding lexical_weights {str: float} to Qdrant-compatible {int: fl, Raised when the GPU runs out of memory during embedding., Wraps FlagEmbedding's BGEM3FlagModel for dense + sparse embedding generation. (+37 more)

### Community 25 - "Community 25"
Cohesion: 0.05
Nodes (50): BaseSettings, Compute MD5 hash of file content for incremental update detection., Settings, Initialize the health monitor.          Args:             settings: Application, Document, Path, QdrantClient, Settings (+42 more)

### Community 26 - "Community 26"
Cohesion: 0.05
Nodes (33): embedder(), embed_texts splits texts into chunks of _effective_batch., _encode_batch returns dense + sparse for each input text., _encode_batch should raise OOMError when GPU runs out of memory., On repeated OOM, falls back to serial single-text calls., After one OOM, batch size is halved and retry succeeds., _convert_sparse converts lexical_weights {str: float} to {int: float} via MD5., _convert_sparse on empty dict returns empty dict. (+25 more)

### Community 27 - "Community 27"
Cohesion: 0.07
Nodes (36): mock_embedder(), Tests for IngestionPipeline — orchestrates load → chunk → embed → store., Missing files should not be counted as failures (loader skips them)., Unsupported file formats should be skipped, not counted as failures., Sidecar .meta.json should provide tags with highest priority., Parent directory name should be used as fallback tag., CLI tags should be applied when no sidecar exists., Sidecar tags should take priority over CLI tags (merged). (+28 more)

### Community 28 - "Community 28"
Cohesion: 0.08
Nodes (22): FastMCP, create_mcp_app(), MCP server — wires together all components and exposes query_knowledge_base tool, Start the MCP server with health monitoring.      Args:         settings: Applic, Build and configure the FastMCP application.      Wires together all components:, run_mcp_server(), Settings, _patch_heavy_components() (+14 more)

### Community 29 - "Community 29"
Cohesion: 0.06
Nodes (29): embedder(), query_engine(), Tests for QueryEngine — orchestrates embed → hybrid search → rerank., Query passes filter_source and filter_tags to hybrid_search., Settings with test defaults., Hybrid search uses HYBRID_CANDIDATE_K; rerank uses top_k from QueryInput., When reranker fails, results still returned (graceful degradation)., Mock FlagEmbeddingEmbedder. (+21 more)

### Community 30 - "Community 30"
Cohesion: 0.07
Nodes (27): Tests for Reranker using FlagReranker from FlagEmbedding., When compute_score raises an exception, reranker returns candidates unchanged., Settings with CPU device for tests., rerank wraps synchronous compute_score in asyncio.to_thread., _resolve_device returns 'cuda' when available., _resolve_device returns 'cpu' when CUDA is not available., _resolve_device returns explicit device when not 'auto'., Reranker uses fp16=True on CUDA device. (+19 more)

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (22): _build_pipeline(), _build_query_engine(), cleanup_orphans(), cli(), config(), config_reset_batch_size(), config_show(), _get_settings() (+14 more)

### Community 32 - "Community 32"
Cohesion: 0.09
Nodes (22): integration_setup(), qdrant_available(), End-to-end integration test: ingest a markdown file and query it.  Requires: - Q, Create a test RTOS document., Ingest a markdown document and verify it can be queried with real embeddings., Re-ingesting the same file should skip it., Changed file should be re-ingested., Orphan cleanup should remove vectors for deleted files. (+14 more)

### Community 33 - "Community 33"
Cohesion: 0.09
Nodes (12): Exception, Tests for `kh status`., TestStatus, Tests for HealthMonitor (mocked)., Mock QdrantClient for testing., _probe_qdrant should return True when Qdrant responds., _probe_qdrant should return False on exception., _probe_gpu should return (bool, int) tuple. (+4 more)

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (15): HealthMonitor, QueryEngine, Orchestrates the full query flow: embed → hybrid search → rerank.      Depends o, HealthMonitor, Probe GPU availability and free memory via nvidia-smi.          Returns:, Background health prober for Qdrant and GPU.      No Ollama probing — FlagEmbedd, Start the background probe loop.          Args:             interval_seconds: In, Get the current cached health status.          If not yet cached, probes once an (+7 more)

### Community 35 - "Community 35"
Cohesion: 0.10
Nodes (11): Tests for the knowledge-hub CLI.  Uses Click's CliRunner for testing CLI command, Tests for `kh config reset-batch-size`., Tests for `kh cleanup-orphans`., Tests for `kh index`., Tests for `kh serve`., Tests for `kh config show`., TestCleanupOrphans, TestConfigResetBatchSize (+3 more)

### Community 36 - "Community 36"
Cohesion: 0.27
Nodes (15): BaseModel, ChunkResult, QueryInput, QueryResult, QueryInput, QueryResult, Reranker, Execute a query: embed → hybrid search → rerank → build result.          Args: (+7 more)

### Community 37 - "Community 37"
Cohesion: 0.14
Nodes (8): query_knowledge_base should pass filter_source and filter_tags., Tests for create_tools() and the query_knowledge_base function., When model_loaded=False, should return error dict without calling engine., When qdrant=False, should return error dict without calling engine., query_knowledge_base should use default top_k=5., create_tools should return a dict with 'query_knowledge_base' key., query_knowledge_base should call QueryEngine.query with correct params., TestCreateTools

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
Cohesion: 0.40
Nodes (3): Initialize the reranker with the configured model.          Args:             se, Resolve 'auto' to 'cuda' if available, else 'cpu'.          Args:             em, Settings

## Knowledge Gaps
- **124 isolated node(s):** `knowledge-hub`, `Status: COMPLETE`, `Summary`, `Files Changed`, `Test Results` (+119 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 25` to `Community 32`, `Community 33`, `Community 34`, `Community 2`, `Community 36`, `Community 37`, `Community 6`, `Community 44`, `Community 24`, `Community 26`, `Community 27`, `Community 28`, `Community 29`, `Community 30`?**
  _High betweenness centrality (0.256) - this node is a cross-community bridge._
- **Why does `TestCreateTools` connect `Community 37` to `Community 25`, `Community 2`, `Community 36`, `Community 34`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `QueryInput` connect `Community 36` to `Community 32`, `Community 34`, `Community 37`, `Community 6`, `Community 24`, `Community 29`, `Community 31`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `Settings` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`Settings` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `FlagEmbeddingEmbedder` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`FlagEmbeddingEmbedder` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `QdrantVectorStore` (e.g. with `DocumentLoader` and `FastMCP`) actually correct?**
  _`QdrantVectorStore` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `QueryInput` (e.g. with `HealthMonitor` and `IngestionPipeline`) actually correct?**
  _`QueryInput` has 14 INFERRED edges - model-reasoned connections that need verification._