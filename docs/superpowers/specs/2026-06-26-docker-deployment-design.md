# Docker 容器化部署设计

> 2026-06-26 · Status: approved

## Overview

为 knowledge-hub 构建 Docker 镜像，支持 GPU/CPU 双模式，数据库和上传文件通过 volume 持久化保存。以 docker-compose 为主推部署方案。

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base image | `nvidia/cuda:12.4.0-runtime-ubuntu22.04` (GPU) / `python:3.12-slim` (CPU) | GPU 推理需求 vs 通用可移植性 |
| 构建系统 | `uv` (已有 `uv.lock`) | 可复现构建，与开发环境一致 |
| 模型分发 | Volume 挂载，不打包进镜像 | 镜像尺寸可控（~15GB → ~500MB），模型可跨容器复用 |
| 部署方式 | `docker-compose.yml` 为主，`docker run` 为辅 | 标准化 volume/ports/GPU 配置，一键启动 |
| 持久化 | 3 个 volume：storage、data、models | 容器重启/重建数据不丢失 |
| 健康检查 | HTTP 探针 MCP tools/list | 确认模型加载完毕、Qdrant 可用 |

## Architecture

```
┌──────────────────────────────────────────────────┐
│ docker-compose (GPU server)                      │
│                                                  │
│  Container: knowledge-hub                        │
│  ┌────────────────────────────────────────────┐  │
│  │ kh serve --host 0.0.0.0                    │  │
│  │  ├─ MCP Server :8765                       │  │
│  │  └─ HTTP Upload :8766                      │  │
│  │                                            │  │
│  │ AppState (单进程)                           │  │
│  │  ├─ BGE-M3 + BGE-reranker-v2-m3 (GPU)     │  │
│  │  ├─ Qdrant (embedded)                      │  │
│  │  └─ JobManager (async upload)              │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Volumes:                                        │
│  ./storage → /app/storage  (Qdrant data)         │
│  ./data    → /app/data     (uploaded files)      │
│  /data/models/bge-m3 → /models (HF cache)        │
└──────────────────────────────────────────────────┘
```

## File Changes

| File | Change |
|------|--------|
| `Dockerfile` | **New** — 单文件，build arg 切换 GPU/CPU |
| `docker-compose.yml` | **New** — 主推部署方案 |
| `.dockerignore` | **New** — 排除不打包进镜像的文件 |
| `scripts/download-models.sh` | **New** — 模型预下载辅助脚本（可选） |
| `.gitignore` | Remove `uv.lock` so it's tracked |

## 1. Dockerfile

```dockerfile
# Build: docker build -t knowledge-hub:gpu .
#        docker build --build-arg BASE_IMAGE=python:3.12-slim -t knowledge-hub:cpu .
ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Phase 1: Install dependencies only (cached layer, no project install yet)
COPY pyproject.toml uv.lock ./
COPY CLAUDE.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Phase 2: Copy source and install the project itself
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Non-root user
RUN useradd -m -u 1000 kh && chown -R kh:kh /app
USER kh

EXPOSE 8765 8766

# --start-period=120s gives models time to load before health checking begins
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/mcp', data=b'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}', timeout=3)" || exit 1

ENTRYPOINT ["uv", "run", "kh", "serve", "--host", "0.0.0.0"]
```

**关键点**:
- 两阶段构建：`--no-install-project` 只装依赖 → COPY src → 完整 `uv sync` 安装项目。避免 `kh` 命令找不到
- `COPY CLAUDE.md` 解包 hatchling 需要 `readme = "CLAUDE.md"`
- `--start-period=120s` 给模型加载充足预热时间，防止误判 unhealthy 触发重启
- 非 root 用户 `kh` (uid 1000) 运行，提升安全性
- `python3` 替代 `python`（ubuntu 镜像可能没有 `python` 软链）
- `--host 0.0.0.0` 硬编码在 ENTRYPOINT（容器必须绑定 0.0.0.0）

## 2. docker-compose.yml（主推方案）

```yaml
services:
  knowledge-hub:
    image: knowledge-hub:gpu
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
      - ./models/bge-m3:/models
    environment:
      - KH_SERVER_HOST=0.0.0.0
      - KH_QDRANT_PATH=/app/storage/qdrant
      - KH_DATA_DIR=/app/data
      - KH_SERVER_AUTH_TOKEN=${KH_SERVER_AUTH_TOKEN}
      - HF_HOME=/models
    env_file:
      - .env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

**Volume 说明**:

| Volume | 容器路径 | 内容 | 持久化原因 |
|--------|---------|------|-----------|
| `./storage` | `/app/storage` | Qdrant 向量数据 + 元数据 | 数据库本体，丢失=重建全部索引 |
| `./data` | `/app/data` | 上传的文件 + `kh index` 源文件 | 支持重索引 + 文件溯源 |
| `./models/bge-m3` | `/models` | HuggingFace 模型缓存 (~9GB) | 避免每次启动重新下载 |

> 模型路径 `./models/bge-m3` 是相对路径（相对于项目根目录），可按需改为绝对路径。

**环境变量**:

| 变量 | 值 | 说明 |
|------|-----|------|
| `KH_SERVER_HOST` | `0.0.0.0` | 容器内必须绑定 0.0.0.0 |
| `KH_QDRANT_PATH` | `/app/storage/qdrant` | 显式指定嵌入式 Qdrant 路径 |
| `KH_DATA_DIR` | `/app/data` | 上传文件存储目录 |
| `KH_SERVER_AUTH_TOKEN` | `${KH_SERVER_AUTH_TOKEN}` | 从 `.env` 文件传入，绑定 0.0.0.0 必须设置 |
| `HF_HOME` | `/models` | HuggingFace 缓存路径 |

## 3. .dockerignore

```dockerignore
# Python
.venv/
__pycache__/
*.pyc
*.egg-info/

# Git
.git/

# Tests
tests/
.pytest_cache/

# Local data (mounted at runtime, not baked in)
storage/
data/
models/

# Dev artifacts
.superpowers/
.tokenmiser/
graphify-out/
docs/

# IDE
.vscode/
.idea/
.claude/

# Markdown (except CLAUDE.md — required by hatchling build)
*.md
!CLAUDE.md

# Misc
LICENSE
```

## 4. 模型预下载

模型不打包进镜像。首次部署前下载到 volume 目录：

```bash
# 使用标准 HuggingFace 缓存结构（匹配容器内 HF_HOME=/models）
pip install huggingface_hub

export HF_HOME=$(pwd)/models/bge-m3
huggingface-cli download BAAI/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3
```

下载后目录结构：
```
models/bge-m3/
└── hub/
    ├── models--BAAI--bge-m3/
    │   └── snapshots/<hash>/     # ~6.6GB
    └── models--BAAI--bge-reranker-v2-m3/
        └── snapshots/<hash>/     # ~2.2GB
```

FlagEmbedding 和 transformers 在 `HF_HOME=/models` 下自动匹配此结构。**不要用 `--local-dir` 参数**，它会绕开标准缓存路径，导致容器内找不到模型而触发重复下载。

## 5. 构建 & 运行

```bash
# -- 构建 --
# GPU（默认）
docker build -t knowledge-hub:gpu .
# CPU
docker build --build-arg BASE_IMAGE=python:3.12-slim -t knowledge-hub:cpu .

# -- 启动 --
docker compose up -d     # 后台启动
docker compose logs -f   # 查看日志
docker compose down      # 停止

# -- 纯 docker run（不使用 compose） --
docker run -d \
  --gpus all \
  -p 8765:8765 -p 8766:8766 \
  -v ./storage:/app/storage \
  -v ./data:/app/data \
  -v ./models/bge-m3:/models \
  -e KH_SERVER_HOST=0.0.0.0 \
  -e KH_SERVER_AUTH_TOKEN=${KH_SERVER_AUTH_TOKEN} \
  -e KH_QDRANT_PATH=/app/storage/qdrant \
  -e KH_DATA_DIR=/app/data \
  -e HF_HOME=/models \
  --restart unless-stopped \
  knowledge-hub:gpu

# -- 验证 --
curl http://<host>:8765/mcp -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## 6. 健康检查策略

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "..." || exit 1
```

- 每 30s 向 MCP tools/list 发 JSON-RPC 请求
- `--start-period=120s`：启动后前 120 秒不执行健康检查，给模型加载充足的预热时间
- 5s 超时，3 次连续失败 → `unhealthy`
- 首次启动后总计 120 + 30×3 = 210s 容错窗口

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| 模型目录为空 | FlagEmbedding 自动从 HuggingFace 下载 → 慢，但不会崩溃 |
| storage/ 目录为空 | Qdrant 自动创建 collection |
| GPU 不可用（CPU 镜像） | EMBED_DEVICE=auto 自动回退 CPU |
| Qdrant 数据损坏 | 删除 storage/ 重新 `kh index` |

## 8. What Is NOT in Scope

- **多副本 / 高可用** — 嵌入式 Qdrant 不支持多进程并发写，单容器是设计约束
- **HTTPS/TLS** — 由反向代理（nginx/caddy）处理
- **DockerHub 发布** — 首次聚焦本地构建
- **CI/CD 流水线** — 后续迭代
- **模型自动下载 init 容器** — 手动 huggingface-cli 足够，后续可选优化
