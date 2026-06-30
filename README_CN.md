# Knowledge Hub 知识库

**本地优先的向量 RAG 知识库，支持 MCP + HTTP 上传接口。**

Knowledge Hub 可以导入文档（Markdown、PDF、纯文本、HTML），使用 BGE-M3 稠密+稀疏向量嵌入，存储到 Qdrant，并通过混合搜索 + 交叉编码器重排序进行查询 —— 全部本地运行，无需云端 API 调用。外部 Agent 可通过 HTTP 上传文件，通过 MCP 查询知识。

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

- Python 3.12

无需外部服务 —— Qdrant 默认以嵌入式模式运行。

### 安装

```bash
git clone https://github.com/Lee-shihao/knowledge-hub.git && cd knowledge-hub
uv sync

# 激活虚拟环境（可选）
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

```bash
# 拉取镜像
docker pull saxiburry/knowledge-hub:latest

# 运行
docker run -d \
  --name knowledge-hub \
  --gpus all \
  -p 8765:8765 \
  -p 8766:8766 \
  -v kh_data:/app \
  -e KH_SERVER_HOST=0.0.0.0 \
  -e KH_DATA_DIR=/app/data \
  -e KH_STORAGE_DIR=/app/storage \
  -e KH_QDRANT_PATH=/app/storage/qdrant \
  -e HF_HOME=/app/models \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e KH_SERVER_AUTH_TOKEN=your-secret-token \
  saxiburry/knowledge-hub:latest
```

或使用 Docker Compose：

```yaml
services:
  knowledge-hub:
    image: saxiburry/knowledge-hub:latest
    ports:
      - "8765:8765"
      - "8766:8766"
    volumes:
      - kh_data:/app
    environment:
      - KH_SERVER_HOST=0.0.0.0
      - KH_DATA_DIR=/app/data
      - KH_STORAGE_DIR=/app/storage
      - KH_QDRANT_PATH=/app/storage/qdrant
      - HF_HOME=/app/models
      - HF_ENDPOINT=https://hf-mirror.com
      - KH_SERVER_AUTH_TOKEN=${KH_SERVER_AUTH_TOKEN}
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  kh_data:
```

```bash
export KH_SERVER_AUTH_TOKEN=your-secret-token
docker compose up -d
```

> `kh_data` 是命名卷，包含所有持久化数据（文档、Qdrant 索引、模型缓存约 2.2GB），避免重启丢失。

### 构建策略

| 触发条件 | 镜像标签 |
|---------|---------|
| push tag `v*` | `0.1.0`, `latest` |

```bash
git tag v0.2.0
git push origin v0.2.0
```

## MCP 服务器使用

MCP 服务器通过 JSON-RPC（streamable-http 传输）暴露 3 个工具。

| 工具 | 说明 |
|------|------|
| `query_knowledge_base` | 语义搜索，混合稠密+稀疏向量 + 交叉编码器重排序 |
| `list_kb_sources` | 列出所有已索引源文件（包含分块数和内容哈希） |
| `get_kb_status` | 系统健康状态（模型、Qdrant、GPU）+ 集合统计 |

### 远程访问（局域网）

```bash
export KH_SERVER_AUTH_TOKEN=test-token-123
kh serve --host 0.0.0.0
```

```bash
# 查询
curl -s -X POST http://192.168.30.125:8765/mcp \
  -H "Authorization: Bearer test-token-123" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query_knowledge_base","arguments":{"query":"你的问题","top_k":5}}}'
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

通过 HTTP 上传文件即可自动导入。

```
POST /upload                    GET /upload/status/{job_id}
Content-Type: multipart/form    响应：
  file: <二进制文件>（必填）       {
  tags: "tag1,tag2"（可选）         "job_id": "e3ce9f20b6fc",
                                    "filename": "test-upload.md",
响应：                                "status": "done",
  {"job_id": "e3ce9f20b6fc",        "chunks": 1,
   "status": "pending"}             "error": null,
                                    ...}
支持格式：.md .txt .pdf
  .html .htm .docx .rst
```

| 状态 | 含义 |
|------|------|
| `pending` | 任务已排队 |
| `processing` | 导入管道运行中（加载 → 分块 → 嵌入 → 存储） |
| `done` | 索引完成，立即可查询 |
| `failed` | 导入出错（见 `error` 字段） |

上传和 MCP 共用 `KH_SERVER_AUTH_TOKEN`。本地访问（默认 127.0.0.1）无需认证。

## Agent 技能集成

为 AI Agent（Hermes、Claude Code、OpenClaw）安装 Knowledge Hub 技能。技能内置 Python 上传脚本，Agent 直接调用，无需手动写 curl。

```bash
# 一行命令安装技能
curl -fsSL https://raw.githubusercontent.com/Lee-shihao/knowledge-hub/main/install_skill.sh | bash -s -- hermes

# 也支持：claude、openclaw
```

安装后的目录结构：
```
~/.hermes/skills/knowledge-hub/
├── SKILL.md
└── scripts/
    └── upload.py
```

然后配置所需的环境变量：

```bash
echo 'KNOWLEDGE_HUB_BASE_URL="http://<服务器IP>:8766"' >> ~/.hermes/.env
echo 'KNOWLEDGE_HUB_TOKEN="your-token-here"' >> ~/.hermes/.env
```

| 变量 | 必填 | 说明 |
|------|------|------|
| `KNOWLEDGE_HUB_BASE_URL` | 是 | 上传服务器 HTTP 端点（8766 端口） |
| `KNOWLEDGE_HUB_TOKEN` | 是 | 认证令牌，需与服务端的 `KH_SERVER_AUTH_TOKEN` 一致 |

> 这些变量是给 Agent 技能使用的，而非服务器端。令牌值必须与服务端配置的 `KH_SERVER_AUTH_TOKEN` 一致。

## 许可证

Apache-2.0
