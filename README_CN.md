# Knowledge Hub 知识库

**本地优先的向量 RAG 知识库，支持 MCP + HTTP 上传接口。**

Knowledge Hub 可以导入文档（Markdown、PDF、纯文本、HTML），使用 BGE-M3 稠密+稀疏向量嵌入，存储到 Qdrant，并通过混合搜索 + 交叉编码器重排序进行查询 —— 全部本地运行，无需云端 API 调用。外部 Agent 可通过 HTTP 上传文件，通过 MCP 查询知识。

## 架构

```
GPU 服务器 — kh serve 单进程双服务
┌─────────────────────────────────────────────────────────┐
│  anyio task group                                       │
│                                                         │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ uvicorn :8765     │  │ uvicorn :8766              │  │
│  │ MCP 服务器         │  │ HTTP 上传服务器             │  │
│  │                   │  │                            │  │
│  │ query_kb          │  │ POST /upload               │  │
│  │ list_sources      │  │ GET  /upload/status/{id}   │  │
│  │ get_status        │  │                            │  │
│  └────────┬──────────┘  └─────────────┬──────────────┘  │
│           │                           │                 │
│           └───────────┬───────────────┘                 │
│                       ▼                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ AppState（共享状态）                               │   │
│  │  - embedder（BGE-M3，GPU，只加载一份）             │   │
│  │  - reranker（BGE-reranker-v2-m3）                 │   │
│  │  - pipeline → 导入管道                            │   │
│  │  - job_manager → 异步上传任务                      │   │
│  │  - query_engine → 混合搜索 + 重排序               │   │
│  │  - qdrant_client → 嵌入式 Qdrant                  │   │
│  │    （默认 ./storage/qdrant/）                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

  外部 Agent（Claude Code / Herd / Hermes）
    → 上传文件：HTTP POST :8766/upload
    → 查询知识：MCP :8765
```

## 功能特性

- **混合搜索**：稠密向量（BGE-M3）+ 稀疏向量（词法权重）通过倒数排名融合（RRF）
- **交叉编码器重排序**：BGE-reranker-v2-m3 对候选结果重新打分，提升精度
- **增量导入**：基于内容哈希跳过未修改文件，文件修改后自动重新导入
- **孤儿清理**：检测并删除已删除源文件的向量
- **嵌入式 Qdrant**：无需外部数据库，Qdrant 进程内运行（默认），可选外部模式
- **HTTP 上传服务**：`POST /upload`（multipart/form-data），异步任务轮询 `GET /upload/status/{id}`
- **MCP 服务器**：3 个工具 — `query_knowledge_base`、`list_kb_sources`、`get_kb_status` — 基于 FastMCP（streamable-http），可选认证 + IP 过滤
- **CLI 命令行**：通过 `kh` 命令完全控制 —— 导入、查询、状态、配置、服务
- **CPU/GPU 自动切换**：FlagEmbedding 自动检测 CUDA；无 GPU 时优雅降级到 CPU
- **OOM 容错**：CUDA 显存不足时自动减小批大小，通过 `kh config reset-batch-size` 重置

## 快速开始

### 前置条件

- Python 3.12+

无需外部服务 —— Qdrant 默认以嵌入式模式运行。

### 安装

```bash
# 克隆并安装
git clone https://github.com/Lee-shihao/knowledge-hub.git && cd knowledge-hub
uv sync

# 激活虚拟环境（可选，用于直接使用 kh 命令）
source .venv/bin/activate

# 或通过 uv run 运行（无需激活）
uv run kh --help

# 首次运行会下载模型（约 2.2GB）
kh index --path ./data
```

### 使用方法

```bash
# ---- 导入 ----
kh index --path ./my-docs
kh index --path ./my-docs --tags "python,ml"  # 带标签
kh index --force                              # 强制重新导入所有文件

# ---- 查询 ----
kh query "优先级继承是如何工作的？"
kh query "调度算法" -k 10                       # 返回前 10 条结果

# ---- 管理 ----
kh status                                     # 集合统计
kh cleanup-orphans                            # 清理已删除文件的向量
kh config show                                # 查看当前配置
kh config reset-batch-size                    # 重置嵌入批大小

# ---- 服务 ----
kh serve                                      # MCP (:8765) + HTTP 上传 (:8766)
kh serve --no-upload                          # 仅 MCP
kh serve --host 0.0.0.0 --port 8765 --upload-port 8766
```

### 通过 HTTP 上传文件

```bash
# 上传文件（本地无需认证）
curl -X POST http://127.0.0.1:8766/upload \
  -F "file=@my-doc.pdf" \
  -F "tags=research,ml"

# 响应：{"job_id":"abc123def456","status":"pending"}

# 轮询任务状态
curl http://127.0.0.1:8766/upload/status/abc123def456

# 响应：{"job_id":"...","filename":"my-doc.pdf","status":"done","chunks":15,...}
```

### 环境变量

所有配置使用 `KH_` 前缀，可通过以下方式配置：

1. **环境变量**（推荐用于部署）：
   ```bash
   export KH_EMBED_DEVICE=cuda          # 使用 GPU（自动启用 fp16）
   kh index --path ./data
   ```

2. **`.env` 文件**（推荐用于开发）：
   ```bash
   # 在项目根目录创建 .env 文件
   cat > .env << 'EOF'
   KH_EMBED_DEVICE=cpu               # 强制使用 CPU（禁用 GPU，使用 fp32）
   KH_CHUNK_MAX_TOKENS=512
   KH_HYBRID_CANDIDATE_K=30
   EOF

   kh config show  # 验证配置
   ```

3. **CLI 参数覆盖**（用于一次性修改）：
   ```bash
   kh serve --host 0.0.0.0 --port 8765 --upload-port 8766
   ```

> **提示**：`KH_EMBED_DEVICE` 控制嵌入/重排序模型的运行位置：
> - `auto` — 自动检测 CUDA，无 GPU 时回退到 CPU（默认）
> - `cuda` — 强制使用 GPU，自动启用 fp16 加速推理
> - `cpu` — 强制使用 CPU，使用 fp32（较慢但无需 GPU）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KH_SERVER_HOST` | `127.0.0.1` | MCP 和上传服务器的绑定地址 |
| `KH_MCP_PORT` | `8765` | MCP 服务器端口 |
| `KH_UPLOAD_PORT` | `8766` | HTTP 上传服务器端口 |
| `KH_UPLOAD_ENABLED` | `true` | 启用 HTTP 上传服务器 |
| `KH_SERVER_AUTH_TOKEN` | — | MCP 和上传的认证令牌（绑定非本机地址时必填） |
| `KH_SERVER_ALLOWED_IPS` | `[]` | MCP 服务器 IP 白名单 |
| `KH_EMBED_MODEL` | `BAAI/bge-m3` | 嵌入模型 HuggingFace ID |
| `KH_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型 HuggingFace ID |
| `KH_EMBED_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KH_QDRANT_MODE` | `embedded` | Qdrant 模式：`embedded`（本地存储）或 `http`（外部服务器） |
| `KH_QDRANT_PATH` | `./storage/qdrant` | 嵌入式 Qdrant 数据目录 |
| `KH_QDRANT_URL` | `http://localhost:6333` | 外部 Qdrant 端点（`QDRANT_MODE=http` 时使用） |
| `KH_QDRANT_COLLECTION` | `knowledge_hub` | 集合名称 |
| `KH_CHUNK_MAX_TOKENS` | `512` | 每个分块的最大 token 数 |
| `KH_CHUNK_OVERLAP` | `0.1` | 分块之间的重叠比例 |
| `KH_EMBED_BATCH_SIZE` | `16` | 嵌入批大小 |
| `KH_MAX_FILE_SIZE_MB` | `200` | 上传文件最大大小 |
| `KH_HYBRID_CANDIDATE_K` | `20` | 重排序前获取的候选数 |
| `KH_FINAL_TOP_K` | `5` | 重排序后返回的最终结果数 |
| `KH_DATA_DIR` | `./data` | 文档源目录 |
| `KH_STORAGE_DIR` | `./storage` | 元数据存储目录 |

## Docker 部署

不想折腾 Python 环境的用户可以直接用 Docker 运行。

### 拉取镜像

```bash
# 最新稳定版
docker pull saxiburry/knowledge-hub:latest

# 或指定版本
docker pull saxiburry/knowledge-hub:0.1.0
```

### Docker Compose（推荐）

创建 `docker-compose.yml`：

```yaml
services:
  knowledge-hub:
    image: saxiburry/knowledge-hub:latest
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - ./data:/app/data
      - ./storage:/app/storage
      - kh_models:/app/models
    environment:
      - KH_EMBED_DEVICE=cpu
      - KH_SERVER_HOST=0.0.0.0
      - KH_DATA_DIR=/app/data
      - KH_STORAGE_DIR=/app/storage
      - KH_QDRANT_PATH=/app/storage/qdrant
      - HF_HOME=/app/models
      - HF_ENDPOINT=https://hf-mirror.com
    restart: unless-stopped

volumes:
  kh_models:
```

```bash
# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

> `kh_models` 是命名卷，保存 HuggingFace 模型缓存（约 2.2GB），避免每次重启重新下载。

### 使用

```bash
# 上传文件
curl -X POST http://localhost:8766/upload -F "file=@my-doc.md"

# 查询
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"你的问题","top_k":5}}}'
```

### GPU 支持

```yaml
environment:
  - KH_EMBED_DEVICE=cuda
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 构建策略

| 触发条件 | 镜像标签 | 用途 |
|---------|---------|------|
| push 到 main | `sha-<短哈希>` | CI 验证，开发调试 |
| push tag `v*` | `0.1.0`, `latest`, `0.1` | 正式发布 |

版本号通过 git tag 管理，手动打 tag 时才发布稳定版：

```bash
git tag v0.2.0
git push origin v0.2.0
```

## MCP 服务器使用

MCP 服务器通过 JSON-RPC（streamable-http 传输）暴露 3 个工具，可直接用 `curl` 调用。

### MCP 工具

| 工具 | 说明 |
|------|------|
| `query_knowledge_base` | 语义搜索，混合稠密+稀疏向量 + 交叉编码器重排序 |
| `list_kb_sources` | 列出所有已索引源文件（包含分块数和内容哈希） |
| `get_kb_status` | 系统健康状态（模型、Qdrant、GPU）+ 集合统计 |

### 远程访问（局域网）

绑定非 localhost 地址需要设置认证令牌（MCP 和上传共用）：

```bash
# 启动服务
export KH_SERVER_AUTH_TOKEN=test-token-123
kh serve --host 0.0.0.0
# [info] server_starting mcp=http://0.0.0.0:8765/mcp upload=http://0.0.0.0:8766/upload
```

**列出可用工具：**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# 返回 3 个工具：query_knowledge_base, list_kb_sources, get_kb_status
```

**查询知识库：**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"BCM2835 SPI接口数量","top_k":3}}}'

# 返回 3 条结果（查询耗时 743ms）：
#   Top 1（score 7.19）：BCM2835 提供 2 个 SPI 接口：SPI0（标准，2 个片选）和 SPI1（辅助，3 个片选）
#     — 来源 test-upload.md，章节 "SPI 接口数量"
#   Top 2（score 5.29）：bcm2835-arm-peripherals.pdf 第 152 页 — 芯片数据手册
#   Top 3（score 3.63）：test-upload.md 概述部分
```

**列出已索引源文件：**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_kb_sources"}}'

# 返回：{"sources":[{"filename":"bcm2835-arm-peripherals.pdf","chunk_count":2560,...}],"count":1}
```

**检查系统状态：**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_kb_status"}}'

# 返回：
#   model_loaded: true        qdrant: true        gpu_available: true
#   gpu_memory_free_mb: 14393 collection: knowledge_hub
#   total_chunks: 2564        total_sources: 1
```

### 本地访问（无需认证）

```bash
kh serve
# 在 127.0.0.1 上暴露相同端点，无需 Authorization 头
```

### 在 AI 客户端中配置

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://<服务器IP>:8765/mcp",
      "transport": "streamable-http",
      "headers": {"Authorization": "Bearer your-secret-token"}
    }
  }
}
```

## HTTP 上传服务

通过 HTTP 上传文件即可自动导入。服务端校验格式后保存到数据目录，异步索引。

### 完整流程（在 GPU 服务器上已验证）

**1. 上传文件：**

```bash
$ curl -s -X POST http://192.168.30.125:8766/upload \
    -H "Authorization: Bearer test-token-123" \
    -F "file=@test-upload.md" \
    -F "tags=spi,bcm2835,test"

{"job_id":"e3ce9f20b6fc","status":"pending"}
```

**2. 轮询任务状态（小文件约 1 秒完成）：**

```bash
$ curl -s http://192.168.30.125:8766/upload/status/e3ce9f20b6fc \
    -H "Authorization: Bearer test-token-123"

{
  "job_id": "e3ce9f20b6fc",
  "filename": "test-upload.md",
  "status": "done",
  "chunks": 1,
  "error": null,
  "created_at": "2026-06-25T10:38:36.562323+00:00",
  "completed_at": "2026-06-25T10:38:37.563172+00:00",
  "failed_files": []
}
```

**3. 通过 MCP 查询刚上传的内容：**

```bash
$ curl -s -X POST http://192.168.30.125:8765/mcp \
    -H "Authorization: Bearer test-token-123" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"BCM2835 SPI接口数量","top_k":3}}}'

# 最高分结果（7.19）即为刚上传的文件内容
```

### API 参考

```
POST /upload                    GET /upload/status/{job_id}
Content-Type: multipart/form    响应：
  file: <二进制文件>（必填）       {
  tags: "tag1,tag2"（可选）         "job_id": "e3ce9f20b6fc",
                                    "filename": "test-upload.md",
响应：                                "status": "done",
  {"job_id": "e3ce9f20b6fc",        "chunks": 1,
   "status": "pending"}             "error": null,
                                    "created_at": "2026-06-25T10:38:36",
支持格式：.md .txt .pdf             "completed_at": "2026-06-25T10:38:37",
  .html .htm .docx .rst             "failed_files": []
                                  }
```

| 状态 | 含义 |
|------|------|
| `pending` | 任务已排队 |
| `processing` | 导入管道运行中（加载 → 分块 → 嵌入 → 存储） |
| `done` | 索引完成，立即可查询 |
| `failed` | 导入出错（见 `error` 字段） |

上传和 MCP 共用 `KH_SERVER_AUTH_TOKEN`。本地访问（默认 127.0.0.1）无需认证。

## 项目结构

```
src/knowledge_hub/
├── config.py              # 配置（pydantic-settings，KH_ 环境前缀）
├── schemas.py             # 数据模式：ChunkMetadata, DocumentChunk, QueryInput, QueryResult
├── cli/
│   └── main.py            # Click CLI：index, query, status, cleanup-orphans, config, serve
├── ingestion/
│   ├── chunker.py         # SemanticChunker — 标题感知分块
│   ├── embedder.py        # FlagEmbeddingEmbedder — BGE-M3 稠密+稀疏嵌入
│   ├── loaders.py         # DocumentLoader — .md/.txt/.pdf 加载与哈希计算
│   └── pipeline.py        # IngestionPipeline — 加载→分块→嵌入→存储
├── retrieval/
│   ├── query_engine.py    # QueryEngine — 嵌入→混合搜索→重排序
│   └── reranker.py        # Reranker — FlagReranker 重排序，失败时优雅降级
├── server/
│   ├── app_state.py       # AppState — MCP + 上传服务共享组件注入
│   ├── health.py          # HealthMonitor — Qdrant + GPU 后台健康探测
│   ├── job_manager.py     # JobManager — 异步上传任务追踪与串行化
│   ├── mcp_server.py      # FastMCP 应用组装，支持认证 + IP 过滤
│   ├── tools.py           # MCP 工具：query_knowledge_base, list_kb_sources, get_kb_status
│   └── upload_server.py   # HTTP 上传应用 — POST /upload, GET /upload/status/{id}
└── storage/
    ├── metadata.py        # SourceMetadataManager — 哈希追踪，孤儿清理
    └── vector_store.py    # QdrantVectorStore — 混合搜索，插入，删除
```

## 测试

```bash
# 单元测试（无需外部服务）
pytest -m "not integration"

# 集成测试（需要 Qdrant，设置 KH_QDRANT_MODE=embedded 或 localhost:6333）
pytest tests/test_integration.py -v -s

# 所有测试
pytest
```

| 测试套件 | 数量 | 依赖 |
|---------|------|------|
| 单元测试 | ~175 | 无（使用 mock） |
| 集成测试 | ~7 | Qdrant + FlagEmbedding 模型（约 2.2GB 下载） |

## 依赖

| 包 | 版本 | 用途 |
|---|------|------|
| FlagEmbedding | 1.4.0 | BGE-M3 嵌入 + BGE-reranker-v2-m3 重排序 |
| transformers | ≥4.40, <5.0 | FlagReranker 分词器（5.x 移除了 `prepare_for_model`） |
| qdrant-client | ≥1.12.0 | 向量存储 + 混合搜索（支持嵌入式模式） |
| fastmcp | ≥2.3.0 | MCP 服务器框架 |
| llama-index | ≥0.12.0 | 文档读取器 |
| click | ≥8.0 | CLI 框架 |
| starlette | * | HTTP 上传服务器 |
| uvicorn | * | MCP + 上传服务的 ASGI 服务器 |
| anyio | * | 双服务启动的任务组 |
| structlog | ≥24.0 | 结构化日志 |
| pydantic | ≥2.0 | 模式验证 |
| pydantic-settings | ≥2.0 | 环境变量配置 |

## 许可证

Apache-2.0
