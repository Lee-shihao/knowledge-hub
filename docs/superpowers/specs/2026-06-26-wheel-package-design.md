# Wheel Package + Docker Simplification

> 2026-06-26 · Status: approved

## Overview

将 knowledge-hub 构建为标准 Python wheel 包，Docker 镜像通过 `pip install` 安装。同时将 Python 最低版本从 3.12 降至 3.10，扩大兼容范围。

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 构建方式 | `uv build --wheel` → 标准 whl 包 | 可分发、可 pip install、Docker 内无 symlink 问题 |
| Docker 安装 | `uv pip install --system dist/*.whl` | console_scripts 直接生成 `kh` 命令，无需 uv run |
| Python 版本 | `>=3.10` | 代码只用到 3.10 特性（`str \| None`），无 3.11+/3.12+ 依赖 |
| 外部分发 | GitHub Release 发布 whl | 最小可行分发，不引入 PyPI 注册流程 |
| Docker 基础镜像 | `nvidia/cuda:12.4.0-runtime-ubuntu22.04` | Ubuntu 22.04 自带 Python 3.10，满足最低版本 |
| Docker 运行时 | `python3` + 系统级 pip install | 无 venv symlink、无 uv run、无权限问题 |

## Architecture

```
┌───────────────────────────────────────────┐
│ Builder Stage                             │
│  nvidia/cuda:12.4.0-runtime-ubuntu22.04   │
│  COPY pyproject.toml uv.lock CLAUDE.md    │
│  COPY src/                                │
│  RUN uv build --wheel                     │
│  → dist/knowledge_hub-0.1.0-py3-none-any.whl │
└───────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│ Runtime Stage                             │
│  nvidia/cuda:12.4.0-runtime-ubuntu22.04   │
│  apt install python3 libgomp1             │
│  COPY --from=builder dist/*.whl /tmp/     │
│  RUN uv pip install --system /tmp/*.whl   │
│  → kh 命令可用（console_scripts）          │
│  ENTRYPOINT ["kh", "serve", "--host", ...] │
└───────────────────────────────────────────┘

外部安装:
  pip install https://github.com/<user>/knowledge-hub/releases/download/v0.1.0/knowledge_hub-0.1.0-py3-none-any.whl
```

## File Changes

| File | Change |
|------|--------|
| `pyproject.toml` | `requires-python = ">=3.10"`（原 `>=3.12`） |
| `uv.lock` | 重新 `uv lock` 解析 3.10 兼容依赖树 |
| `Dockerfile` | 重写：`uv build` → `uv pip install --system` → 直接 `kh` entrypoint |
| `docker-compose.yml` | 不变 |
| `.dockerignore` | 新增 `dist/` 排除 |

## 1. Python 3.12 → 3.10 兼容性分析

**代码中只用到 3.10 特性**：

| 语法 | PEP | 最低版本 | 使用位置 |
|------|-----|---------|---------|
| `str \| None` | 604 | 3.10 | tools.py, vector_store.py, schemas.py, pipeline.py, metadata.py |
| `list[str]` | 585 | 3.9 | 全代码 |
| `from __future__ import annotations` | 563 | 3.7 | app_state.py |

**未使用的 3.11+ / 3.12+ 特性**：
- `match/case`（3.10）
- `except*`（3.11）
- `type X = ...`（PEP 695, 3.12）
- `Self` 类型（3.11）

**依赖库兼容性**：torch、transformers、FlagEmbedding、fastmcp、qdrant-client、pydantic 均支持 3.10+。

**改动**：`pyproject.toml` 1 行修改 + 重新 `uv lock`。

## 2. Wheel 构建

### pyproject.toml（关键字段）

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "knowledge-hub"
version = "0.1.0"
requires-python = ">=3.10"

[project.scripts]
kh = "knowledge_hub.cli.main:cli"
```

`uv build --wheel` 产出 `dist/knowledge_hub-0.1.0-py3-none-any.whl`。

## 3. Dockerfile

```dockerfile
# Build:
#   docker build -t knowledge-hub .
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel

# ============================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/dist/*.whl /tmp/

# System-level install — creates /usr/local/bin/kh with correct shebang
RUN uv pip install --system /tmp/knowledge_hub-0.1.0-py3-none-any.whl \
    && rm /tmp/*.whl

RUN useradd -m -u 1000 kh
USER kh

EXPOSE 8765 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["kh", "serve", "--host", "0.0.0.0"]
```

**对比旧 Dockerfile 的关键改进**：

| 旧问题 | 新方案 |
|--------|--------|
| `uv sync` editable install, 无 console_scripts | `pip install` 直接生成 `kh` 命令 |
| venv symlink 跨阶段断裂 | 系统级安装，无 symlink |
| 需要 `uv run --no-sync` | 直接 `kh serve` |
| 需要复制 uv Python store | 不需要，回到系统 python3 |
| permission denied (root home 700) | 系统级路径，权限天然正确 |

## 4. 构建 & 运行

```bash
# 构建镜像
docker build -t knowledge-hub .

# 启动
docker compose up -d

# 构建 wheel（本地分发包）
uv build --wheel
ls dist/knowledge_hub-0.1.0-py3-none-any.whl

# 外部安装
pip install dist/knowledge_hub-0.1.0-py3-none-any.whl
kh serve --host 0.0.0.0
```

## 5. GitHub Release 发布流程（后续）

```bash
# CI 中
uv build --wheel
gh release create v0.1.0 dist/*.whl
# 用户安装
pip install https://github.com/<user>/knowledge-hub/releases/download/v0.1.0/knowledge_hub-0.1.0-py3-none-any.whl
```

当前聚焦 Docker，GitHub Release 留待后续 CI/CD。

## 6. What Is NOT in Scope

- PyPI 发布
- CI/CD GitHub Actions 流水线
- 多 Python 版本 CI 测试矩阵
