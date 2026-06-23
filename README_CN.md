# Knowledge Hub 知识库

**本地优先的向量 RAG 知识库，支持 MCP 接口。**

Knowledge Hub 可以导入文档（Markdown、纯文本），使用 BGE-M3 稠密+稀疏向量嵌入，存储到 Qdrant，并通过混合搜索 + 交叉编码器重排序进行查询 —— 全部本地运行，无需云端 API 调用。

## 架构

```
┌──────────┐    ┌──────────────────────────────────────────────┐
│  CLI/MCP │───▶│  QueryEngine 查询引擎                        │
│  服务器   │    │  嵌入 → 混合搜索(稠密+稀疏) → 重排序         │
└──────────┘    └──────────────────────────────────────────────┘
      │                         │              │
      ▼                         ▼              ▼
┌──────────┐            ┌────────────┐  ┌─────────────┐
│ Ingestion│            │ Qdrant      │  │ FlagReranker │
│ 导入管道  │            │ 向量存储    │  │ 重排序器      │
└──────────┘            └────────────┘  └─────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│ 加载 → 分块 → 嵌入 → 存储            │
│ .md/.txt  语义   BGE-M3  Qdrant      │
│          分块器  (稠密+   +元数据     │
│                  稀疏)    存储       │
└─────────────────────────────────────┘
```

## 功能特性

- **混合搜索**：稠密向量（BGE-M3）+ 稀疏向量（词法权重）通过倒数排名融合（RRF）
- **交叉编码器重排序**：BGE-reranker-v2-m3 对候选结果重新打分，提升精度
- **增量导入**：基于内容哈希跳过未修改文件，文件修改后自动重新导入
- **孤儿清理**：检测并删除已删除源文件的向量
- **MCP 服务器**：通过 FastMCP（SSE 传输）暴露 `query_knowledge_base` 工具，支持可选认证 + IP 过滤
- **CLI 命令行**：通过 `kh` 命令完全控制 —— 导入、查询、状态、配置、服务
- **CPU/GPU 自动切换**：FlagEmbedding 自动检测 CUDA；无 GPU 时优雅降级到 CPU
- **OOM 容错**：CUDA 显存不足时自动减小批大小，通过 `kh config reset-batch-size` 重置

## 快速开始

### 前置条件

- Python 3.12+
- [Qdrant](https://qdrant.tech/) 运行在 localhost:6333

```bash
# 启动 Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 安装

```bash
# 克隆并安装
git clone <repo-url> && cd knowledge-hub
uv sync

# 首次运行会下载模型（约 2.2GB）
kh index --path ./data
```

### 使用方法

```bash
# 导入文档
kh index --path ./my-docs
kh index --path ./my-docs --tags "python,ml"  # 带标签
kh index --force                              # 强制重新导入所有文件

# 查询
kh query "优先级继承是如何工作的？"
kh query "调度算法" -k 10                       # 返回前 10 条结果

# 查看状态
kh status

# 清理已删除文件
kh cleanup-orphans

# 配置
kh config show
kh config reset-batch-size

# 启动 MCP 服务器
kh serve
kh serve --host 0.0.0.0 --port 9999
```

### 环境变量

所有配置使用 `KH_` 前缀（或 `.env` 文件）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KH_MCP_HOST` | `127.0.0.1` | MCP 服务器绑定地址 |
| `KH_MCP_PORT` | `8765` | MCP 服务器端口 |
| `KH_MCP_AUTH_TOKEN` | — | 认证令牌（绑定非本机地址时必填） |
| `KH_MCP_ALLOWED_IPS` | `[]` | MCP 服务器 IP 白名单 |
| `KH_EMBED_MODEL` | `BAAI/bge-m3` | 嵌入模型 HuggingFace ID |
| `KH_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排序模型 HuggingFace ID |
| `KH_EMBED_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KH_QDRANT_URL` | `http://localhost:6333` | Qdrant 端点 |
| `KH_QDRANT_COLLECTION` | `knowledge_hub` | 集合名称 |
| `KH_CHUNK_MAX_TOKENS` | `512` | 每个分块的最大 token 数 |
| `KH_CHUNK_OVERLAP` | `0.1` | 分块之间的重叠比例 |
| `KH_EMBED_BATCH_SIZE` | `16` | 嵌入批大小 |
| `KH_HYBRID_CANDIDATE_K` | `20` | 重排序前获取的候选数 |
| `KH_FINAL_TOP_K` | `5` | 重排序后返回的最终结果数 |
| `KH_DATA_DIR` | `./data` | 文档源目录 |
| `KH_STORAGE_DIR` | `./storage` | 元数据存储目录 |

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
│   ├── loaders.py         # DocumentLoader — .md/.txt 加载与哈希计算
│   └── pipeline.py        # IngestionPipeline — 加载→分块→嵌入→存储
├── retrieval/
│   ├── query_engine.py    # QueryEngine — 嵌入→混合搜索→重排序
│   └── reranker.py        # Reranker — FlagReranker 重排序，失败时优雅降级
├── server/
│   ├── health.py          # HealthMonitor — Qdrant + GPU 后台健康探测
│   ├── mcp_server.py      # FastMCP 应用组装，支持认证 + IP 过滤
│   └── tools.py           # MCP 工具：query_knowledge_base
└── storage/
    ├── metadata.py        # SourceMetadataManager — 哈希追踪，孤儿清理
    └── vector_store.py    # QdrantVectorStore — 混合搜索，插入，删除
```

## 测试

```bash
# 单元测试（无需外部服务）
pytest -m "not integration"

# 集成测试（需要 Qdrant 运行在 localhost:6333）
pytest tests/test_integration.py -v -s

# 所有测试
pytest
```

| 测试套件 | 数量 | 依赖 |
|---------|------|------|
| 单元测试 | 126 | 无（使用 mock） |
| 集成测试 | 7 | Qdrant + FlagEmbedding 模型（约 2.2GB 下载） |

## 依赖

| 包 | 版本 | 用途 |
|---|------|------|
| FlagEmbedding | 1.4.0 | BGE-M3 嵌入 + BGE-reranker-v2-m3 重排序 |
| transformers | ≥4.40, <5.0 | FlagReranker 分词器（5.x 移除了 `prepare_for_model`） |
| qdrant-client | ≥1.12.0 | 向量存储 + 混合搜索 |
| fastmcp | ≥2.3.0 | MCP 服务器框架 |
| llama-index | ≥0.12.0 | 文档读取器 |
| click | ≥8.0 | CLI 框架 |
| structlog | ≥24.0 | 结构化日志 |
| pydantic | ≥2.0 | 模式验证 |
| pydantic-settings | ≥2.0 | 环境变量配置 |

## 许可证

Apache-2.0
