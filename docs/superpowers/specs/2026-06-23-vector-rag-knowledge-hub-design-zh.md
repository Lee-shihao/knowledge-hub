# Vector RAG 知识中心 — 设计规格

日期：2026-06-23 | 状态：设计已批准

## 1. 概述

为技术文档（数据手册、协议规范、技术手册、书籍）构建一个**本地优先**的 Vector RAG 知识中心。该系统摄取多种格式的文档，将其索引到向量数据库中，并通过 MCP Server 暴露检索接口，以便 Claude Code 和其他兼容 MCP 的 Agent 可以自主查询知识库。

**目标环境**：局域网部署、单 GPU 机器，支持多个 Agent 客户端。

## 2. 技术选型

| 层级 | 选择 | 理由 |
|------|------|------|
| RAG 框架 | **LlamaIndex** | 具备原生文档处理流水线，在此用例中比 LangChain 提供更好的索引抽象 |
| 嵌入模型 | **bge-m3**（通过 Ollama） | 稠密+稀疏双输出，多语言支持，1024 维，本地 GPU |
| 向量存储 | **Qdrant**（本地） | 原生稀疏向量支持、混合搜索 API、Rust 性能 |
| 重排序器 | **bge-reranker**（通过 Ollama） | 交叉编码器，本地 GPU |
| MCP 框架 | **FastMCP 2.x**（纯 Python） | 内置 Bearer 认证，支持 Streamable HTTP，异步原生 |
| MCP 传输 | **SSE**（主要）、**Streamable HTTP**（备用） | 目前使用 SSE；可通过配置切换 |
| 配置 | **pydantic-settings** | 通过 .env / 环境变量驱动，类型安全 |
| 数据模型 | **pydantic** | 在 CLI / MCP Server 间共享 schema |
| 构建系统 | **uv + pyproject.toml** | 现代 Python 工具链 |
| 测试 | **pytest + testcontainers** | 在集成测试中使用真实的 Qdrant/Ollama 容器 |

## 3. 项目结构

```
knowledge-hub/
├── docs/superpowers/specs/          # 设计文档
├── src/knowledge_hub/
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings，所有环境变量
│   ├── schemas.py                   # DocumentChunk、ChunkMetadata、QueryInput、QueryResult
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # IngestionPipeline 编排器
│   │   ├── loaders.py               # 格式分发的文档加载器
│   │   ├── chunker.py               # 语义分块 + 标题链
│   │   └── embedder.py              # OllamaEmbedder：bge-m3 稠密+稀疏，OOM 感知
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # QdrantVectorStore：CRUD、稀疏向量、孤立清理
│   │   └── metadata.py              # 源文件哈希跟踪，用于增量更新
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_engine.py          # QueryEngine：编排搜索 → 重排序
│   │   └── reranker.py              # bge-reranker 交叉编码器
│   ├── server/
│   │   ├── __init__.py
│   │   ├── mcp_server.py            # 带认证的 FastMCP Server
│   │   ├── tools.py                 # MCP Tool 定义（query_knowledge_base）
│   │   └── health.py                # 运行时健康探针（Ollama、Qdrant、GPU）
│   └── cli/
│       ├── __init__.py
│       └── main.py                  # CLI：index、query、status、delete、config
├── tests/
│   ├── conftest.py                  # QdrantContainer、OllamaContainer 测试夹具
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   ├── test_ingestion_pipeline.py
│   ├── test_mcp_tools.py
│   └── test_integration.py          # 完整摄取 + 查询端到端测试
├── data/                            # 源文档
├── storage/                         # Qdrant 持久化 + batch_size 状态
├── pyproject.toml
└── CLAUDE.md
```

## 4. 组件架构

```
┌──────────────────────────────────────────────┐
│  MCP Server (FastMCP)  │  CLI (click/typer)  │  ← 呈现层
│  - Bearer Token 认证   │  - index/query/     │
│  - 运行时健康检查       │    status/delete    │
│  - 支持 Streamable HTTP │    config           │
├──────────────────────────────────────────────┤
│            QueryEngine (异步)                │  ← 检索层
│  嵌入查询 → 混合搜索 → 重排序                │
├──────────────────────────────────────────────┤
│  IngestionPipeline  │  QdrantVectorStore     │  ← 摄取 + 存储层
│  加载→分块→嵌入     │  CRUD + 孤立清理       │
├──────────────────────────────────────────────┤
│   LlamaIndex  │  Qdrant Client  │  Ollama    │  ← 基础设施
└──────────────────────────────────────────────┘
```

每个组件职责单一：
- **IngestionPipeline** — 负责将文件转为索引向量，不关心查询
- **QueryEngine** — 负责搜索和重排序，不关心文档如何进入
- **QdrantVectorStore** — 拥有所有 Qdrant I/O，是稀疏向量转换的唯一位置
- **MCP Server / CLI** — 薄壳层，负责将组件连接起来

## 5. 数据模型（schemas.py）

```python
from pydantic import BaseModel, Field
from datetime import datetime

class ChunkMetadata(BaseModel):
    source_file: str
    source_hash: str                        # 文件内容的 md5，用于增量更新
    page_number: int | None = None
    heading_path: list[str] = []            # ["第 3 章", "3.2 任务调度", "3.2.1 优先级继承"]
    tags: list[str] = []
    ingested_at: datetime

class DocumentChunk(BaseModel):
    id: str                                 # md5(source_file + "|".join(heading_path) + text[:200])，确定性生成
    text: str
    dense_embedding: list[float] = Field(exclude=True)   # 1024 维，不包含在日志/序列化中
    sparse_embedding: dict[int, float] = Field(exclude=True)  # 在 vector_store 中转为 SparseVector
    metadata: ChunkMetadata

class QueryInput(BaseModel):
    query: str
    top_k: int = 5
    filter_source: str | None = None
    filter_tags: list[str] | None = None

class ChunkResult(BaseModel):
    text: str
    source_file: str
    page_or_section: str
    heading_path: list[str]
    score: float

class QueryResult(BaseModel):
    results: list[ChunkResult]
    query_time_ms: float
```

## 6. 核心数据流

### 6.1 摄取流水线

```
原始文件（data/*.pdf、*.md、*.html、*.docx、*.txt）
    │
    ▼
DocLoader ─── 格式检测 → SimpleDirectoryReader 分发
    │         PDF → Markdown 转换（保留表格/代码块）
    │         Markdown/HTML/Word → 直接解析
    │
    ▼ List[Document]
Chunker ─── 语义分割：优先使用 Markdown 标题 / 章节标记
    │       max_tokens = 512（硬限制，超大表格/代码块会被分割）
    │       overlay = 0.1（块间 10% 重叠）
    │       保留 heading_path 链以提供上下文
    │
    ▼ List[DocumentChunk]（稠密/稀疏嵌入尚未填充）
Embedder ─── Ollama / bge-m3 → {dense: 1024d float[], sparse: dict[int, float]}
    │       批量处理 + OOM 感知自动降级
    │       每个批次最多 3 次重试 + 指数退避
    │       所有批次失败时回退到单文本处理
    │
    ▼ List[DocumentChunk]（嵌入已填充）
QdrantVectorStore ─── dict[int,float] → SparseVector(indices=[...], values=[...])
    │                 带元数据的 upsert
    │                 孤立清理：删除本地文件系统中已不存在的 source_file 对应的向量
    │
    ▼ 完成 — 输出摄取报告：{total, succeeded, failed, skipped, orphans_cleaned}
```

**元数据丰富（标签赋值）**：
标签按以下优先级合并：
1. **侧边 `.meta.json` 文件** — 与文档并列放置，例如 `data/rtos-book/.meta.json` 内容为 `{"tags": ["rtos", "embedded", "c"]}`。优先级最高。
2. **目录名推断** — 父目录名自动作为标签（例如 `data/datasheets/` → 标签 `datasheets`）。可通过配置禁用。
3. **CLI 覆盖** — `kh index --tags "tag1,tag2"` 对本次摄取的所有文档应用标签。

标签存储在 `ChunkMetadata.tags` 中，并持久化到 Qdrant payload 以支持服务端过滤。

**增量更新逻辑**：
1. 对每个本地文件计算 `md5(file_content)`
2. 在 Qdrant 中维护一个轻量级的 `source_metadata` 集合（独立于向量点），以 `source_file` 为键，存储 `{source_hash, last_ingested_at, chunk_count}`
3. 比较本地哈希与 `source_metadata` → 跳过哈希匹配的文件；重新摄取变更的文件
4. 摄取完成后，删除本地已不存在的 source_file 对应的向量点和 `source_metadata` 条目（孤立清理）

### 6.2 查询流水线

```
用户查询："FreeRTOS 如何实现优先级继承？"
    │
    ▼
Embedder ─── 查询文本 → {dense: 1024d, sparse: dict[int, float]}
    │
    ▼
Qdrant ─── 混合搜索 + RRF（Reciprocal Rank Fusion）
    │      服务端完成稠密 + 稀疏融合
    │      top_k = 20 个候选
    │      可选：按 source_file / tags 过滤
    │
    ▼ top 20
Reranker ─── bge-reranker 交叉编码器对 20 个结果重排序
    │        返回 top_k = 5
    │
    ▼
QueryResult ─── [{text, source_file, page, heading_path, score}]
```

为什么选择 RRF 而非加权融合：Qdrant 原生 RRF 无需手动调参，通常能达到或超过加权融合的效果。未来 Qdrant 支持加权查询时可配置切换。

## 7. 配置（config.py）

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KH_", env_file=".env")

    # 网络
    MCP_HOST: str = "127.0.0.1"           # 仅当设置 MCP_AUTH_TOKEN 时才允许 "0.0.0.0"
    MCP_PORT: int = 8765
    MCP_TRANSPORT: Literal["sse", "streamable-http"] = "sse"

    # 认证（局域网部署必须设置）
    MCP_AUTH_TOKEN: str | None = None      # None → 强制 MCP_HOST="127.0.0.1"
    MCP_ALLOWED_IPS: list[str] = []

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "bge-m3"
    RERANK_MODEL: str = "bge-reranker"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_hub"

    # 摄取
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP: float = 0.1
    EMBED_BATCH_SIZE: int = 16           # 初始值；OOM 时自动降级并持久化
    MAX_FILE_SIZE_MB: int = 200
    WARN_FILE_SIZE_MB: int = 50

    # 查询
    HYBRID_CANDIDATE_K: int = 20
    FINAL_TOP_K: int = 5

    # 存储路径（相对于项目根目录或绝对路径）
    DATA_DIR: str = "./data"
    STORAGE_DIR: str = "./storage"
```

## 8. GPU OOM 处理（embedder.py）

```python
import asyncio
import json
from pathlib import Path

class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._effective_batch = self._load_persisted_batch_size()

    async def embed_batch(self, texts: list[str]) -> list[dict]:
        for attempt in range(3):
            try:
                return await self._call_ollama(texts, self._effective_batch)
            except OOMError:
                self._effective_batch = max(4, self._effective_batch // 2)
                self._persist_batch_size()
                await asyncio.sleep(2 ** attempt)
                continue
        # 所有批次尝试失败 → 单文本串行回退
        results = []
        for t in texts:
            results.extend(await self._call_ollama([t], 1))
        return results

    def _persist_batch_size(self):
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        state_file.write_text(json.dumps({"batch_size": self._effective_batch}))

    def _load_persisted_batch_size(self) -> int:
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())["batch_size"]
        return self.settings.EMBED_BATCH_SIZE
```

**恢复**：降级后的 batch_size 会跨重启持久化。执行 `kh config reset-batch-size` 可重置为默认值。不使用自动增大策略 — OOM 通常表示真实的显存限制。

## 9. MCP Server 安全

```python
# server/mcp_server.py
from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier

def create_mcp_server(settings: Settings, query_engine: QueryEngine) -> FastMCP:
    if settings.MCP_AUTH_TOKEN:
        verifier = StaticTokenVerifier(settings.MCP_AUTH_TOKEN)
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if settings.MCP_HOST != "127.0.0.1":
            raise ValueError("未设置 MCP_AUTH_TOKEN 时，MCP_HOST 必须为 127.0.0.1")
        mcp = FastMCP("knowledge-hub")

    # IP 白名单中间件（认证后应用）
    if settings.MCP_ALLOWED_IPS:
        from starlette.middleware.base import BaseHTTPMiddleware
        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                client_ip = request.client.host
                if client_ip not in settings.MCP_ALLOWED_IPS:
                    from starlette.responses import JSONResponse
                    return JSONResponse({"error": "Forbidden"}, status_code=403)
                return await call_next(request)
        mcp.add_middleware(IPAllowlistMiddleware)

    mcp.add_tool(query_knowledge_base, name="query_knowledge_base")
    return mcp
```

**局域网部署检查清单**：
1. 设置 `KH_MCP_AUTH_TOKEN` 为强随机值
2. 设置 `KH_MCP_HOST=0.0.0.0`（仅在有 Token 时允许）
3. 优先使用 Tailscale/WireGuard 隧道，而非直接暴露 TCP
4. 如使用反向代理，建议在代理层增加 IP 白名单

## 10. 运行时健康检查（server/health.py）

```python
import asyncio
from dataclasses import dataclass

@dataclass
class HealthStatus:
    ollama: bool
    qdrant: bool
    gpu_available: bool
    gpu_memory_free_mb: int

class HealthMonitor:
    def __init__(self, settings: Settings, qdrant_client, ollama_client):
        self.settings = settings
        self._qdrant = qdrant_client
        self._ollama = ollama_client
        self._cached_status: HealthStatus | None = None
        self._task: asyncio.Task | None = None

    async def start(self, interval_seconds: int = 30):
        self._task = asyncio.create_task(self._probe_loop(interval_seconds))

    async def get_status(self) -> HealthStatus:
        if self._cached_status is None:
            self._cached_status = await self._probe_all()
        return self._cached_status

    async def _probe_loop(self, interval: int):
        while True:
            self._cached_status = await self._probe_all()
            await asyncio.sleep(interval)
```

QueryEngine 在调用 Ollama/Qdrant 前会检查 `HealthMonitor.get_status()`。若不健康，则返回清晰错误而非超时。

## 11. 错误处理策略

| 层级 | 策略 |
|------|------|
| MCP Server | 捕获所有异常，返回结构化错误 JSON，绝不暴露栈追踪。遵守 MCP 错误协议。 |
| Query Engine | Ollama 超时 → 重试（最多 2 次）。Qdrant 不可达 → “知识库不可用”。集合为空 → “知识库为空，请先摄取文档”。重排序器失败 → 优雅降级：返回 Qdrant 原始混合搜索结果（top_k），并在 query_time_ms 元数据中附加警告。 |
| Ingestion Pipeline | 单个文件失败不中断整个批次。失败文件记录在报告中。嵌入失败 → 指数退避重试（最多 3 次），然后记录日志并跳过。 |
| 全局 | 使用 structlog 结构化日志。级别：DEBUG（开发）、INFO（运维）、WARNING（降级）、ERROR（故障）。 |

**边缘情况**：

| 场景 | 行为 |
|------|------|
| 启动时 Ollama 未运行 | 快速失败并给出清晰错误信息和启动指引 |
| 查询时 Qdrant 为空 | 返回空结果并附带提示消息 |
| 文件 > 50MB | 记录警告，继续处理 |
| 文件 > 200MB | 拒绝并报错 |
| 并发查询 | 异步 MCP Server + Qdrant 原生并发读支持 |
| 重复文件摄取 | source_hash 匹配 → 跳过 |
| 孤立向量（源文件已删除） | 摄取后执行清理 |
| 嵌入时 GPU OOM | 自动降级 batch_size 并持久化，回退到串行 |
| 重排序器不可用/超时 | 直接返回 Qdrant 的 top_k 结果（未重排序），附加警告 |

## 12. 测试策略

| 级别 | 测试内容 | 方法 |
|------|----------|------|
| 单元 | 分块逻辑、嵌入格式、哈希计算、schema 验证 | pytest |
| 集成 | Qdrant CRUD、完整摄取→查询流程、MCP Tool 调用 | pytest + testcontainers（QdrantContainer、OllamaContainer） |
| 契约 | MCP Tool 输入/输出 schema 符合性 | pydantic 验证器 |
| 手动 | CLI 命令、真实 Ollama 模型、Claude Code 的 MCP 连接 | 文档化的手动测试步骤 |

### testcontainers 测试夹具（tests/conftest.py）

```python
import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer
from testcontainers.ollama import OllamaContainer

@pytest.fixture(scope="session")
def qdrant():
    with QdrantContainer("qdrant/qdrant:1.12.4") as qc:
        yield QdrantClient(
            host=qc.get_container_host_ip(),
            port=qc.get_exposed_port(6333)
        )

@pytest.fixture(scope="session")
def ollama():
    # 注意：首次运行会拉取 bge-m3（约 2GB）。CI 中可预缓存模型层。
    with OllamaContainer() as oc:
        yield oc.get_container_host_ip(), oc.get_exposed_port(11434)
```

## 13. MCP Tool 契约

```json
// Tool: query_knowledge_base
// 输入 schema：
{
  "query": "string (必填) - 自然语言查询",
  "top_k": "int (可选，默认 5) - 返回结果数量",
  "filter_source": "string | null (可选) - 按源文件名过滤",
  "filter_tags": "list[string] | null (可选) - 按标签过滤"
}

// 输出 schema：
{
  "results": [
    {
      "text": "string",
      "source_file": "string",
      "page_or_section": "string",
      "heading_path": ["string", ...],
      "score": "float"
    }
  ],
  "query_time_ms": "float"
}
```

Claude Code 通过以下 MCP 配置连接：

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://<host>:8765/sse",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

## 14. CLI 命令

```
kh index                  # 摄取 data/ 目录下的所有文档
kh index --path <dir>     # 从指定目录摄取
kh index --tags "a,b"     # 添加标签（会被 .meta.json 覆盖）
kh index --force          # 强制重新摄取，忽略 source_hash 缓存
kh query "<问题>"         # 直接查询（无需 MCP）
kh query "<q>" -k 10      # 指定 top_k
kh status                 # 显示文档数量、最近摄取时间、存储大小
kh cleanup-orphans        # 手动触发孤立向量清理
kh config reset-batch-size  # 重置 OOM 降级后的 batch_size
kh config show            # 显示当前生效配置
```

说明：`--tags` 应用于本次运行的所有文件，但优先级最低 —— 侧边 `.meta.json` 和目录名标签优先。这是故意的：`--tags` 是针对未整理目录的快速方式；每个文档的元数据应放在 `.meta.json` 中。

## 15. 本次范围外（不在本规格中）

- 多用户认证（单局域网信任域）
- 通过 API 更新文档（仅通过 CLI / 文件系统摄取）
- 聊天/对话式界面（仅支持单次检索）
- 知识图谱集成（graphify 为独立项目）
- 文档级访问控制
- 监控仪表盘 / Prometheus 指标
- `kh index --meta <file>` 批量元数据注入（使用侧边 `.meta.json` 文件代替；CLI `--tags` 覆盖简单场景）

---

*设计规格结束。*