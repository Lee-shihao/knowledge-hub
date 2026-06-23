# Graph Report - knowledge-hub  (2026-06-23)

## Corpus Check
- 30 files · ~15,936 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 150 nodes · 147 edges · 24 communities (17 shown, 7 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1d0c777e`
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

## God Nodes (most connected - your core abstractions)
1. `Vector RAG 知识中心 — 设计规格` - 16 edges
2. `Vector RAG Knowledge Hub — Design Spec` - 16 edges
3. `Global Constraints` - 15 edges
4. `SourceMetadataManager` - 11 edges
5. `Settings` - 8 edges
6. `Task 2 Report: Config + Schemas` - 8 edges
7. `Task 1 Report` - 6 edges
8. `ChunkMetadata` - 5 edges
9. `Codebase Index` - 4 edges
10. `DocumentChunk` - 3 edges

## Surprising Connections (you probably didn't know these)
- `test_settings_defaults()` --calls--> `Settings`  [EXTRACTED]
  tests/test_config.py → src/knowledge_hub/config.py
- `test_settings_from_env()` --calls--> `Settings`  [EXTRACTED]
  tests/test_config.py → src/knowledge_hub/config.py
- `settings()` --calls--> `Settings`  [EXTRACTED]
  tests/test_metadata.py → src/knowledge_hub/config.py
- `test_chunk_metadata_creation()` --calls--> `ChunkMetadata`  [EXTRACTED]
  tests/test_schemas.py → src/knowledge_hub/schemas.py
- `test_chunk_metadata_defaults()` --calls--> `ChunkMetadata`  [EXTRACTED]
  tests/test_schemas.py → src/knowledge_hub/schemas.py

## Import Cycles
- None detected.

## Communities (24 total, 7 thin omitted)

### Community 0 - "Project Documentation & Tooling"
Cohesion: 0.50
Nodes (4): Apache 2.0 License, CLAUDE.md Project Guidance File, graphify Knowledge Graph Tool, knowledge-hub Knowledge Management System

### Community 1 - "Community 1"
Cohesion: 0.33
Nodes (5): CLAUDE.md, Codebase Index, Config, Docs, Source

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (10): BaseSettings, Settings, QdrantClient, Settings, Manages source file metadata in a separate Qdrant collection.      Uses a lightw, SourceMetadataManager, test_settings_defaults(), test_settings_from_env() (+2 more)

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
Cohesion: 0.29
Nodes (11): BaseModel, ChunkMetadata, ChunkResult, DocumentChunk, QueryInput, QueryResult, test_chunk_metadata_creation(), test_chunk_metadata_defaults() (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (8): Commits, Concerns, Deviations from Brief, Files Created/Modified, Status: DONE, Task 2 Report: Config + Schemas, TDD Steps Followed, Test Summary

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (6): Commits, Concerns, Self-Review, Task 1 Report, Tests, What was implemented

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (4): Codebase Navigation, Current State, graphify, Project Overview

## Knowledge Gaps
- **75 isolated node(s):** `knowledge-hub`, `Tasks`, `Task 1: Project Scaffolding`, `What was implemented`, `Tests` (+70 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Are the 2 inferred relationships involving `SourceMetadataManager` (e.g. with `Settings` and `metadata_mgr()`) actually correct?**
  _`SourceMetadataManager` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Settings` (e.g. with `QdrantClient` and `Settings`) actually correct?**
  _`Settings` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `knowledge-hub`, `Manages source file metadata in a separate Qdrant collection.      Uses a lightw`, `Temporary storage directory for Qdrant and batch_size state.` to the rest of the system?**
  _77 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.1076923076923077 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `Community 5` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._