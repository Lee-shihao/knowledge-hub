# CLAUDE.md

## Codebase Navigation
**Always read `CODEBASE_INDEX.md` before opening any source file.**
It contains the complete file map with exports and purpose for every file.
Use it to locate the exact file you need, then read only that file.


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

knowledge-hub — Local-first Vector RAG knowledge base with MCP + HTTP upload interface.

License: Apache 2.0

## Current State

knowledge-hub is a functioning Local-first Vector RAG knowledge base with MCP + HTTP upload interface. See `README_CN.md` for full documentation.

## Version Management

**Single source of truth:** `version` field in `pyproject.toml`. Currently `0.1.0`.

### How version is read at runtime

`importlib.metadata.version("knowledge-hub")` — reads from installed package metadata. Falls back to `"0.1.0"` if the package is not installed (e.g., raw source checkout).

### Release workflow

```
1. Bump version in pyproject.toml     (e.g., 0.1.0 → 0.2.0)
2. git commit -m "release: v0.2.0"
3. git tag v0.2.0                     (must be valid semver: v<major>.<minor>.<patch>)
4. git push origin dev --tags
```

When code is pushed to GitHub:

| Trigger | Image Tags |
|---------|-----------|
| push to main | `latest` |
| push tag `v*` | `0.2.0`, `latest` |

### Rules

- **Do NOT** create a separate `_version.py` or `VERSION` file. pyproject.toml is the only source.
- **Do NOT** auto-increment versions in CI. Version bumps require human judgment (semver).
- **Do** keep the Dockerfile's `COPY dist/knowledge_hub-*.whl` glob (no hardcoded version).
- Docker image: `saxiburry/knowledge-hub` on Docker Hub.
- CI workflow: `.github/workflows/docker-image.yml`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

尽量使用当前工程下的虚拟python环境，避免污染系统python环境
