# Vector RAG Knowledge Hub — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first Vector RAG knowledge hub — ingest mixed-format technical documents, index them into Qdrant via LlamaIndex + bge-m3 embeddings, expose retrieval as an MCP tool for Claude Code.

**Architecture:** Layered Python package — ingestion pipeline (load→chunk→embed→store), retrieval engine (hybrid search→rerank), MCP server (FastMCP with Bearer auth), CLI (click). Each layer has clean interfaces and can be tested independently.

**Tech Stack:** Python 3.12+, uv, LlamaIndex, Qdrant 1.12+, FlagEmbedding (bge-m3, bge-reranker-v2-m3), FastMCP 2.x, pydantic-settings, structlog, pytest + testcontainers

## Global Constraints

- Python version: >=3.12
- Build system: uv + pyproject.toml (PEP 621)
- All file paths under `src/knowledge_hub/`, tests under `tests/`
- Async throughout (embedding, Qdrant I/O, MCP handlers)
- TDD: write the test first, verify it fails, implement, verify it passes, commit
- Config via pydantic-settings with `KH_` env prefix
- structlog for all logging
- pydantic models for all data contracts (schemas.py)
- Bearer token auth required when MCP_HOST != 127.0.0.1
- Reranker failure → graceful degradation (return un-reranked results)
- GPU OOM → auto-degrade batch_size, persist, manual reset via CLI
- **Virtual environment**: All `uv` and `python`/`pytest` commands MUST be run inside the project's virtual environment at `.venv/`. Activate with `source .venv/bin/activate` before running any Python command, or prefix commands with `.venv/bin/python` and `.venv/bin/pytest`.
- **FlagEmbedding (not Ollama)**: The project uses FlagEmbedding's `BGEM3FlagModel` and `FlagReranker` directly, NOT Ollama. The embedder class is `FlagEmbeddingEmbedder` (not `OllamaEmbedder`). There is no `OLLAMA_BASE_URL` config field. Models auto-download from HuggingFace on first use.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/knowledge_hub/__init__.py`
- Create: `src/knowledge_hub/ingestion/__init__.py`
- Create: `src/knowledge_hub/storage/__init__.py`
- Create: `src/knowledge_hub/retrieval/__init__.py`
- Create: `src/knowledge_hub/server/__init__.py`
- Create: `src/knowledge_hub/cli/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable package `knowledge-hub`, test infrastructure with Qdrant/Ollama fixtures

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "knowledge-hub"
version = "0.1.0"
description = "Local-first Vector RAG knowledge hub with MCP interface"
readme = "CLAUDE.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.12"
dependencies = [
    "llama-index>=0.12.0",
    "llama-index-readers-file>=0.3.0",
    "llama-index-embeddings-ollama>=0.4.0",
    "qdrant-client>=1.12.0",
    "fastmcp>=2.3.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "structlog>=24.0",
    "click>=8.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "testcontainers>=4.0",
]

[project.scripts]
kh = "knowledge_hub.cli.main:cli"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create all __init__.py files**

```bash
mkdir -p src/knowledge_hub/ingestion src/knowledge_hub/storage src/knowledge_hub/retrieval src/knowledge_hub/server src/knowledge_hub/cli tests
touch src/knowledge_hub/__init__.py
touch src/knowledge_hub/ingestion/__init__.py
touch src/knowledge_hub/storage/__init__.py
touch src/knowledge_hub/retrieval/__init__.py
touch src/knowledge_hub/server/__init__.py
touch src/knowledge_hub/cli/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py with testcontainers fixtures**

```python
import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Temporary storage directory for Qdrant and batch_size state."""
    d = tmp_path / "storage"
    d.mkdir()
    return d
```

- [ ] **Step 4: Install the package and verify**

Run: `uv pip install -e ".[dev]"`

Expected: package installs without errors.

- [ ] **Step 5: Verify CLI entry point stubs**

Run: `kh --help`

Expected: error about missing `cli` module (not yet implemented, but confirms entry point is registered).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: scaffold project structure and dependencies"
```

---

### Task 2: Config (config.py) + Schemas (schemas.py)

**Files:**
- Create: `src/knowledge_hub/config.py`
- Create: `src/knowledge_hub/schemas.py`
- Create: `tests/test_config.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings` class (pydantic-settings, `KH_` prefix), `DocumentChunk`, `ChunkMetadata`, `QueryInput`, `ChunkResult`, `QueryResult` pydantic models

- [ ] **Step 1: Write failing test for config**

```python
# tests/test_config.py
import os
from knowledge_hub.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.MCP_HOST == "127.0.0.1"
    assert settings.MCP_PORT == 8765
    assert settings.MCP_TRANSPORT == "sse"
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.EMBED_MODEL == "bge-m3"
    assert settings.RERANK_MODEL == "bge-reranker"
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.QDRANT_COLLECTION == "knowledge_hub"
    assert settings.CHUNK_MAX_TOKENS == 512
    assert settings.CHUNK_OVERLAP == 0.1
    assert settings.EMBED_BATCH_SIZE == 16
    assert settings.MAX_FILE_SIZE_MB == 200
    assert settings.WARN_FILE_SIZE_MB == 50
    assert settings.HYBRID_CANDIDATE_K == 20
    assert settings.FINAL_TOP_K == 5
    assert settings.MCP_AUTH_TOKEN is None
    assert settings.MCP_ALLOWED_IPS == []
    assert settings.DATA_DIR == "./data"
    assert settings.STORAGE_DIR == "./storage"


def test_settings_from_env():
    os.environ["KH_MCP_HOST"] = "0.0.0.0"
    os.environ["KH_MCP_AUTH_TOKEN"] = "test-token"
    settings = Settings()
    assert settings.MCP_HOST == "0.0.0.0"
    assert settings.MCP_AUTH_TOKEN == "test-token"
    del os.environ["KH_MCP_HOST"]
    del os.environ["KH_MCP_AUTH_TOKEN"]
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: ImportError (config module doesn't exist yet).

- [ ] **Step 3: Implement config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KH_", env_file=".env")

    # Network
    MCP_HOST: str = "127.0.0.1"
    MCP_PORT: int = 8765
    MCP_TRANSPORT: Literal["sse", "streamable-http"] = "sse"

    # Auth (required for LAN deployment)
    MCP_AUTH_TOKEN: str | None = None
    MCP_ALLOWED_IPS: list[str] = []

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "bge-m3"
    RERANK_MODEL: str = "bge-reranker"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_hub"

    # Ingestion
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP: float = 0.1
    EMBED_BATCH_SIZE: int = 16
    MAX_FILE_SIZE_MB: int = 200
    WARN_FILE_SIZE_MB: int = 50

    # Query
    HYBRID_CANDIDATE_K: int = 20
    FINAL_TOP_K: int = 5

    # Storage paths
    DATA_DIR: str = "./data"
    STORAGE_DIR: str = "./storage"
```

- [ ] **Step 4: Write failing test for schemas**

```python
# tests/test_schemas.py
from datetime import datetime
from knowledge_hub.schemas import (
    ChunkMetadata, DocumentChunk, QueryInput, ChunkResult, QueryResult
)


def test_chunk_metadata_creation():
    meta = ChunkMetadata(
        source_file="test.pdf",
        source_hash="abc123",
        page_number=42,
        heading_path=["Chapter 1", "Introduction"],
        tags=["test", "pdf"],
        ingested_at=datetime(2026, 6, 23, 12, 0, 0),
    )
    assert meta.source_file == "test.pdf"
    assert meta.heading_path == ["Chapter 1", "Introduction"]


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(source_file="test.pdf", source_hash="abc123")
    assert meta.page_number is None
    assert meta.heading_path == []
    assert meta.tags == []


def test_document_chunk_excludes_embeddings():
    chunk = DocumentChunk(
        id="test_id",
        text="test text",
        dense_embedding=[0.1] * 1024,
        sparse_embedding={0: 0.5, 10: 0.3},
        metadata=ChunkMetadata(source_file="test.pdf", source_hash="abc"),
    )
    # Dense and sparse embeddings should be excluded from serialization
    dumped = chunk.model_dump()
    assert "dense_embedding" not in dumped
    assert "sparse_embedding" not in dumped
    assert "text" in dumped


def test_query_input_defaults():
    qi = QueryInput(query="test query")
    assert qi.top_k == 5
    assert qi.filter_source is None
    assert qi.filter_tags is None


def test_query_result_structure():
    result = QueryResult(
        results=[
            ChunkResult(
                text="answer text",
                source_file="test.pdf",
                page_or_section="p42",
                heading_path=["Ch1"],
                score=0.95,
            )
        ],
        query_time_ms=123.4,
    )
    assert len(result.results) == 1
    assert result.query_time_ms == 123.4
```

- [ ] **Step 5: Run test to verify failure**

Run: `pytest tests/test_schemas.py -v`
Expected: ImportError (schemas module doesn't exist yet).

- [ ] **Step 6: Implement schemas.py**

```python
from datetime import datetime
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source_file: str
    source_hash: str
    page_number: int | None = None
    heading_path: list[str] = []
    tags: list[str] = []
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    id: str
    text: str
    dense_embedding: list[float] = Field(exclude=True)
    sparse_embedding: dict[int, float] = Field(exclude=True)
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

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_schemas.py -v`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/knowledge_hub/config.py src/knowledge_hub/schemas.py tests/test_config.py tests/test_schemas.py
git commit -m "feat: add config (pydantic-settings) and data models (schemas)"
```

---

### Task 3: Storage — Metadata Manager

**Files:**
- Create: `src/knowledge_hub/storage/metadata.py`
- Create: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`
- Produces: `SourceMetadataManager` class
  - `async get_hash(source_file: str) -> str | None`
  - `async upsert(source_file: str, source_hash: str, chunk_count: int) -> None`
  - `async remove(source_file: str) -> None`
  - `async list_sources() -> list[str]`
  - `async orphan_cleanup(local_source_files: set[str]) -> int`

- [ ] **Step 1: Write failing tests for SourceMetadataManager**

```python
# tests/test_metadata.py
import pytest
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.storage.metadata import SourceMetadataManager


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_knowledge_hub",
        STORAGE_DIR=str(temp_storage_dir),
    )


@pytest.fixture
async def metadata_mgr(settings):
    client = QdrantClient(settings.QDRANT_URL)
    mgr = SourceMetadataManager(settings, client)
    await mgr.ensure_collection()
    yield mgr
    # Cleanup
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.asyncio
async def test_upsert_and_get_hash(metadata_mgr):
    await metadata_mgr.upsert("test.pdf", "abc123", 10)
    h = await metadata_mgr.get_hash("test.pdf")
    assert h == "abc123"


@pytest.mark.asyncio
async def test_get_hash_missing(metadata_mgr):
    h = await metadata_mgr.get_hash("nonexistent.pdf")
    assert h is None


@pytest.mark.asyncio
async def test_list_sources(metadata_mgr):
    await metadata_mgr.upsert("a.pdf", "h1", 5)
    await metadata_mgr.upsert("b.pdf", "h2", 3)
    sources = await metadata_mgr.list_sources()
    assert set(sources) == {"a.pdf", "b.pdf"}


@pytest.mark.asyncio
async def test_remove(metadata_mgr):
    await metadata_mgr.upsert("x.pdf", "h1", 1)
    await metadata_mgr.remove("x.pdf")
    assert await metadata_mgr.get_hash("x.pdf") is None


@pytest.mark.asyncio
async def test_orphan_cleanup(metadata_mgr):
    await metadata_mgr.upsert("keep.pdf", "h1", 5)
    await metadata_mgr.upsert("orphan.pdf", "h2", 3)
    removed = await metadata_mgr.orphan_cleanup({"keep.pdf", "other.pdf"})
    assert removed == 1  # orphan.pdf removed
    assert await metadata_mgr.get_hash("orphan.pdf") is None
    assert await metadata_mgr.get_hash("keep.pdf") == "h1"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_metadata.py -v`
Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 3: Implement SourceMetadataManager**

```python
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    CollectionConfigParams, Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from knowledge_hub.config import Settings


class SourceMetadataManager:
    """Manages source file metadata in a separate Qdrant collection.
    
    Uses a lightweight collection keyed by source_file to track
    content hashes for incremental updates and enable O(1) lookups
    without scanning all vector points.
    """

    def __init__(self, settings: Settings, client: QdrantClient):
        self.settings = settings
        self._collection = f"{settings.QDRANT_COLLECTION}_source_meta"
        self._client = client

    async def ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    async def get_hash(self, source_file: str) -> str | None:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
            limit=1,
        )
        if points:
            return points[0].payload.get("source_hash")
        return None

    async def upsert(self, source_file: str, source_hash: str, chunk_count: int) -> None:
        from qdrant_client.models import PointStruct
        import hashlib
        point_id = int(hashlib.md5(source_file.encode()).hexdigest()[:16], 16) % (2**63)
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload={
                        "source_file": source_file,
                        "source_hash": source_hash,
                        "chunk_count": chunk_count,
                    },
                )
            ],
        )

    async def remove(self, source_file: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )

    async def list_sources(self) -> list[str]:
        points, next_offset = self._client.scroll(
            collection_name=self._collection, limit=100
        )
        sources = [p.payload["source_file"] for p in points]
        while next_offset:
            points, next_offset = self._client.scroll(
                collection_name=self._collection, offset=next_offset, limit=100
            )
            sources.extend(p.payload["source_file"] for p in points)
        return sources

    async def orphan_cleanup(self, local_source_files: set[str]) -> int:
        db_sources = set(await self.list_sources())
        orphans = db_sources - local_source_files
        for orphan in orphans:
            await self.remove(orphan)
        return len(orphans)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metadata.py -v`
Expected: FAIL — tests will hit real Qdrant at localhost. Mark as integration tests.

Update conftest.py to add integration marker, then:

Run: `pytest tests/test_metadata.py -v -m "integration or not integration" --no-header`
Expected: tests hit real Qdrant instance. If Qdrant not running locally, tests fail with connection error (expected for now — these are integration tests requiring Qdrant).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/storage/metadata.py src/knowledge_hub/storage/__init__.py tests/test_metadata.py
git commit -m "feat: add SourceMetadataManager for source hash tracking"
```

---

### Task 4: Storage — QdrantVectorStore

**Files:**
- Create: `src/knowledge_hub/storage/vector_store.py`
- Create: `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`, `DocumentChunk`, `ChunkMetadata` from `schemas.py`, `SourceMetadataManager` from `storage/metadata.py`
- Produces: `QdrantVectorStore` class
  - `async ensure_collection() -> None`
  - `async upsert_chunks(chunks: list[DocumentChunk]) -> None`
  - `async hybrid_search(dense_vec: list[float], sparse_vec: dict[int, float], top_k: int, filter_source: str | None, filter_tags: list[str] | None) -> list[tuple[str, float, dict]]`
  - `async delete_by_source(source_file: str) -> None`
  - `async count() -> int`

- [ ] **Step 1: Write failing tests for QdrantVectorStore**

```python
# tests/test_vector_store.py
import hashlib
from datetime import datetime
import pytest
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.schemas import DocumentChunk, ChunkMetadata
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager


def make_chunk_id(source_file: str, heading_path: list[str], text: str) -> str:
    joined = "|".join(heading_path)
    raw = f"{source_file}_{joined}_{text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_kb_vec",
        STORAGE_DIR=str(temp_storage_dir),
    )


@pytest.fixture
async def vector_store(settings):
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    yield store
    # Cleanup
    client.delete_collection(settings.QDRANT_COLLECTION)
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.asyncio
async def test_ensure_collection(vector_store):
    """Collection should exist without error after ensure_collection."""
    # Second call should be idempotent
    await vector_store.ensure_collection()


@pytest.mark.asyncio
async def test_upsert_and_count(vector_store):
    chunk = DocumentChunk(
        id=make_chunk_id("test.pdf", ["Ch1"], "Some text content for testing purposes"),
        text="Some text content for testing purposes",
        dense_embedding=[0.1] * 1024,
        sparse_embedding={0: 0.5, 10: 0.3, 100: 0.8},
        metadata=ChunkMetadata(
            source_file="test.pdf",
            source_hash="abc123",
            heading_path=["Ch1"],
            tags=["test"],
        ),
    )
    await vector_store.upsert_chunks([chunk])
    c = await vector_store.count()
    assert c == 1


@pytest.mark.asyncio
async def test_upsert_idempotent(vector_store):
    """Upserting with same ID should overwrite, not duplicate."""
    chunk = DocumentChunk(
        id=make_chunk_id("test.pdf", ["Ch1"], "content"),
        text="content",
        dense_embedding=[0.0] * 1024,
        sparse_embedding={0: 1.0},
        metadata=ChunkMetadata(source_file="test.pdf", source_hash="h1"),
    )
    await vector_store.upsert_chunks([chunk])
    await vector_store.upsert_chunks([chunk])
    assert await vector_store.count() == 1


@pytest.mark.asyncio
async def test_delete_by_source(vector_store):
    chunk = DocumentChunk(
        id=make_chunk_id("del.pdf", ["Ch1"], "to delete"),
        text="to delete",
        dense_embedding=[0.0] * 1024,
        sparse_embedding={0: 1.0},
        metadata=ChunkMetadata(source_file="del.pdf", source_hash="h1"),
    )
    await vector_store.upsert_chunks([chunk])
    await vector_store.delete_by_source("del.pdf")
    assert await vector_store.count() == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_vector_store.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement QdrantVectorStore**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseVector, PointStruct, ScoredPoint,
    Filter, FieldCondition, MatchValue, MatchAny,
)
from knowledge_hub.config import Settings
from knowledge_hub.schemas import DocumentChunk
from knowledge_hub.storage.metadata import SourceMetadataManager


class QdrantVectorStore:
    """Manages the Qdrant vector collection for document chunks.

    Stores dense (1024d) and sparse vectors natively in Qdrant,
    enabling server-side hybrid search via RRF.
    """

    def __init__(self, settings: Settings, client: QdrantClient, metadata_mgr: SourceMetadataManager):
        self.settings = settings
        self._collection = settings.QDRANT_COLLECTION
        self._client = client
        self._metadata = metadata_mgr

    async def ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection in collections:
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        points = []
        for chunk in chunks:
            sparse_indices = list(chunk.sparse_embedding.keys())
            sparse_values = [chunk.sparse_embedding[i] for i in sparse_indices]
            points.append(PointStruct(
                id=chunk.id,
                vector={
                    "dense": chunk.dense_embedding,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "text": chunk.text,
                    "source_file": chunk.metadata.source_file,
                    "source_hash": chunk.metadata.source_hash,
                    "page_number": chunk.metadata.page_number,
                    "heading_path": chunk.metadata.heading_path,
                    "tags": chunk.metadata.tags,
                },
            ))
        self._client.upsert(collection_name=self._collection, points=points)

    async def hybrid_search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        top_k: int = 20,
        filter_source: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        from qdrant_client.models import QueryRequest, Prefetch

        # Build optional payload filter
        must_conditions = []
        if filter_source:
            must_conditions.append(FieldCondition(key="source_file", match=MatchValue(value=filter_source)))
        if filter_tags:
            must_conditions.append(FieldCondition(key="tags", match=MatchAny(any=filter_tags)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        sparse_indices = list(sparse_vec.keys())
        sparse_values = [sparse_vec[i] for i in sparse_indices]

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=top_k),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query_filter=query_filter,
            limit=top_k,
        )
        return [
            (p.id, p.score, p.payload)
            for p in results.points
        ]

    async def delete_by_source(self, source_file: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )

    async def count(self) -> int:
        info = self._client.count(collection_name=self._collection)
        return info.count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vector_store.py -v`
Expected: tests pass (requires running Qdrant at localhost).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/storage/vector_store.py tests/test_vector_store.py
git commit -m "feat: add QdrantVectorStore with dense+sparse native storage and RRF hybrid search"
```

---

### Task 5: Ingestion — OllamaEmbedder

**Files:**
- Create: `src/knowledge_hub/ingestion/embedder.py`
- Create: `tests/test_embedder.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`
- Produces: `OllamaEmbedder` class
  - `async embed_texts(texts: list[str]) -> list[dict]` — returns `[{dense: list[float], sparse: dict[int, float]}]`
  - `async embed_query(query: str) -> dict` — single query embedding
  - `async reset_batch_size() -> None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embedder.py
import json
from pathlib import Path
import pytest
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import OllamaEmbedder


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        OLLAMA_BASE_URL="http://localhost:11434",
        EMBED_MODEL="bge-m3",
        STORAGE_DIR=str(temp_storage_dir),
        EMBED_BATCH_SIZE=16,
    )


@pytest.fixture
def embedder(settings):
    return OllamaEmbedder(settings)


@pytest.mark.asyncio
async def test_embed_query_returns_dense_and_sparse(embedder):
    result = await embedder.embed_query("test query")
    assert "dense" in result
    assert "sparse" in result
    assert len(result["dense"]) == 1024
    assert isinstance(result["sparse"], dict)
    assert len(result["sparse"]) > 0


@pytest.mark.asyncio
async def test_embed_texts_batch(embedder):
    texts = ["First text", "Second text", "Third text"]
    results = await embedder.embed_texts(texts)
    assert len(results) == 3
    for r in results:
        assert len(r["dense"]) == 1024
        assert isinstance(r["sparse"], dict)


@pytest.mark.asyncio
async def test_batch_size_persistence(embedder, temp_storage_dir):
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    embedder._effective_batch = 4
    embedder._persist_batch_size()
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["batch_size"] == 4


@pytest.mark.asyncio
async def test_reset_batch_size(embedder, temp_storage_dir):
    embedder._effective_batch = 4
    embedder._persist_batch_size()
    await embedder.reset_batch_size()
    assert embedder._effective_batch == 16
    state_file = Path(temp_storage_dir) / ".batch_size_state.json"
    assert not state_file.exists()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_embedder.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement OllamaEmbedder**

```python
import asyncio
import json
from pathlib import Path
import structlog
import httpx
from knowledge_hub.config import Settings

logger = structlog.get_logger()


class OOMError(Exception):
    """Raised when Ollama returns an out-of-memory error."""
    pass


class OllamaEmbedder:
    """Wraps Ollama's bge-m3 for dense + sparse embedding generation.

    Handles batch processing with OOM-aware auto-degradation:
    - Starts at configured batch_size (default 16)
    - On OOM: halves batch_size (min 4), persists to disk
    - After 3 batch failures: falls back to single-text serial embedding
    - Recovery: manual `kh config reset-batch-size`
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.EMBED_MODEL
        self._effective_batch = self._load_persisted_batch_size()

    async def embed_query(self, query: str) -> dict:
        """Embed a single query text. Returns {dense, sparse}."""
        results = await self._embed_batch_with_retry([query])
        return results[0]

    async def embed_texts(self, texts: list[str]) -> list[dict]:
        """Embed a batch of texts. Each result is {dense, sparse}."""
        all_results = []
        for i in range(0, len(texts), self._effective_batch):
            batch = texts[i:i + self._effective_batch]
            batch_results = await self._embed_batch_with_retry(batch)
            all_results.extend(batch_results)
        return all_results

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[dict]:
        for attempt in range(3):
            try:
                return await self._call_ollama(texts)
            except OOMError:
                self._effective_batch = max(4, self._effective_batch // 2)
                self._persist_batch_size()
                logger.warning(
                    "OOM detected, degraded batch_size",
                    new_batch_size=self._effective_batch,
                    attempt=attempt + 1,
                )
                await asyncio.sleep(2 ** attempt)
                continue

        # All batch attempts failed — serial fallback
        logger.warning("Falling back to single-text serial embedding")
        results = []
        for t in texts:
            results.extend(await self._call_ollama([t]))
        return results

    async def _call_ollama(self, texts: list[str]) -> list[dict]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
            )
            if response.status_code != 200:
                if "out of memory" in response.text.lower():
                    raise OOMError(response.text)
                raise RuntimeError(f"Ollama API error: {response.status_code} {response.text}")
            data = response.json()
            # bge-m3 via Ollama returns "embeddings" as list[list[float]]
            # The sparse representation is extracted from the model's output
            # Ollama's /api/embed doesn't natively return sparse; we extract it
            # by requesting bge-m3's sparse embedding via a separate mechanism.
            # For now, use dense-only and generate a simple sparse bag-of-words.
            return [
                {
                    "dense": emb,
                    "sparse": self._sparse_bow(texts[i]),
                }
                for i, emb in enumerate(data["embeddings"])
            ]

    def _sparse_bow(self, text: str) -> dict[int, float]:
        """Generate a simple bag-of-words sparse vector as fallback.
        
        NOTE: Ollama's /api/embed endpoint currently only returns dense
        embeddings. bge-m3's native sparse vectors (lexical weights per token)
        are not exposed. This BoW fallback provides a functional keyword-match
        signal for hybrid search in the MVP.
        
        Future upgrade path: use a separate sparse encoder (e.g., direct
        llama-cpp-python with bge-m3, or a dedicated sparse model) and
        swap this method.
        """
        import re
        tokens = re.findall(r'\w+', text.lower())
        sparse = {}
        for t in tokens:
            h = hash(t) % 100000
            sparse[h] = sparse.get(h, 0.0) + 1.0
        # Normalize
        max_val = max(sparse.values()) if sparse else 1.0
        return {k: v / max_val for k, v in sparse.items()}

    def _persist_batch_size(self):
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"batch_size": self._effective_batch}))

    def _load_persisted_batch_size(self) -> int:
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())["batch_size"]
        return self.settings.EMBED_BATCH_SIZE

    async def reset_batch_size(self) -> None:
        self._effective_batch = self.settings.EMBED_BATCH_SIZE
        state_file = Path(self.settings.STORAGE_DIR) / ".batch_size_state.json"
        if state_file.exists():
            state_file.unlink()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embedder.py -v`
Expected: tests pass (requires Ollama running at localhost with bge-m3 pulled).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/ingestion/embedder.py tests/test_embedder.py
git commit -m "feat: add OllamaEmbedder with OOM-aware batch degradation"
```

---

### Task 6: Ingestion — Chunker

**Files:**
- Create: `src/knowledge_hub/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`, `DocumentChunk`, `ChunkMetadata` from `schemas.py`
- Produces: `SemanticChunker` class
  - `chunk(documents: list[Document], source_file: str, source_hash: str) -> list[DocumentChunk]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunker.py
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.chunker import SemanticChunker


def make_doc(text):
    """Minimal Document-like object for testing."""
    from llama_index.core.schema import Document
    return Document(text=text, metadata={})


def test_chunker_produces_chunks():
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("This is a test document with some content. " * 20)
    chunks = chunker.chunk([doc], "test.txt", "abc123")
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.source_file == "test.txt"
        assert c.metadata.source_hash == "abc123"
        assert len(c.text) > 0


def test_chunker_short_document():
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("Short text.")
    chunks = chunker.chunk([doc], "short.txt", "hash1")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunker_heading_path_in_metadata():
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("# Chapter 1\n\nContent here. " * 5 + "\n\n## Section 1.1\n\nMore content. " * 5)
    chunks = chunker.chunk([doc], "headings.md", "hash1")
    # At least one chunk should have a heading_path
    heading_chunks = [c for c in chunks if c.metadata.heading_path]
    assert len(heading_chunks) > 0


def test_chunk_id_deterministic():
    settings = Settings()
    chunker = SemanticChunker(settings)
    doc = make_doc("Unique content for ID testing. " * 10)
    chunks1 = chunker.chunk([doc], "same.txt", "h1")
    chunks2 = chunker.chunk([doc], "same.txt", "h1")
    assert chunks1[0].id == chunks2[0].id
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_chunker.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement SemanticChunker**

```python
import hashlib
import re
from llama_index.core.schema import Document
from knowledge_hub.config import Settings
from knowledge_hub.schemas import DocumentChunk, ChunkMetadata


class SemanticChunker:
    """Splits documents into semantic chunks for embedding.

    Strategy:
    1. Split by markdown headings first (preserves heading chain)
    2. Within each section, split by paragraph boundaries
    3. Merge small paragraphs until approaching max_tokens
    4. Hard split at max_tokens for oversized elements (tables, code blocks)
    5. Overlap adjacent chunks by settings.CHUNK_OVERLAP ratio
    """

    def __init__(self, settings: Settings):
        self._max_tokens = settings.CHUNK_MAX_TOKENS
        self._overlap = settings.CHUNK_OVERLAP

    def chunk(
        self, documents: list[Document], source_file: str, source_hash: str
    ) -> list[DocumentChunk]:
        chunks = []
        for doc in documents:
            sections = self._split_by_headings(doc.text)
            for heading_chain, section_text in sections:
                section_chunks = self._split_by_tokens(
                    section_text, heading_chain, source_file, source_hash
                )
                chunks.extend(section_chunks)
        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[list[str], str]]:
        """Split text by markdown headings, tracking the heading chain."""
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        sections = []
        current_headings = []
        last_pos = 0

        for match in heading_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Capture text before this heading
            if last_pos < match.start():
                section_text = text[last_pos:match.start()].strip()
                if section_text:
                    sections.append((list(current_headings), section_text))

            # Update heading chain
            current_headings = current_headings[:level - 1]
            current_headings.append(title)
            last_pos = match.end()

        # Remaining text after last heading
        if last_pos < len(text):
            section_text = text[last_pos:].strip()
            if section_text:
                sections.append((list(current_headings), section_text))

        if not sections:
            sections = [([], text)]

        return sections

    def _split_by_tokens(
        self, text: str, heading_chain: list[str], source_file: str, source_hash: str
    ) -> list[DocumentChunk]:
        """Split text into chunks respecting max_tokens, with overlap."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_texts = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self._estimate_tokens(para)

            # Hard split for oversized single elements
            if para_tokens > self._max_tokens:
                if current_texts:
                    chunks.append(self._make_chunk(
                        "\n\n".join(current_texts), heading_chain, source_file, source_hash
                    ))
                    current_texts = []
                    current_tokens = 0
                # Split oversized paragraph into sub-chunks
                sub_chunks = self._hard_split(para, heading_chain, source_file, source_hash)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + para_tokens > self._max_tokens and current_texts:
                chunks.append(self._make_chunk(
                    "\n\n".join(current_texts), heading_chain, source_file, source_hash
                ))
                # Overlap: keep last paragraph if overlap > 0
                if self._overlap > 0 and len(current_texts) > 0:
                    current_texts = [current_texts[-1]]
                    current_tokens = self._estimate_tokens(current_texts[0])
                else:
                    current_texts = []
                    current_tokens = 0

            current_texts.append(para)
            current_tokens += para_tokens

        if current_texts:
            chunks.append(self._make_chunk(
                "\n\n".join(current_texts), heading_chain, source_file, source_hash
            ))

        return chunks

    def _hard_split(
        self, text: str, heading_chain: list[str], source_file: str, source_hash: str
    ) -> list[DocumentChunk]:
        """Split an oversized text element into max_tokens-sized chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self._max_tokens):
            sub_text = " ".join(words[i:i + self._max_tokens])
            chunks.append(self._make_chunk(sub_text, heading_chain, source_file, source_hash))
        return chunks

    def _make_chunk(
        self, text: str, heading_chain: list[str], source_file: str, source_hash: str
    ) -> DocumentChunk:
        joined_headings = "|".join(heading_chain)
        raw_id = f"{source_file}_{joined_headings}_{text[:200]}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()

        return DocumentChunk(
            id=chunk_id,
            text=text,
            dense_embedding=[],
            sparse_embedding={},
            metadata=ChunkMetadata(
                source_file=source_file,
                source_hash=source_hash,
                heading_path=heading_chain,
            ),
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return max(1, len(text) // 4)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_chunker.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: add SemanticChunker with heading-aware splitting and deterministic IDs"
```

---

### Task 7: Ingestion — Document Loaders

**Files:**
- Create: `src/knowledge_hub/ingestion/loaders.py`
- Create: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`
- Produces: `DocumentLoader` class
  - `load_files(paths: list[Path]) -> list[Document]`
  - `compute_hash(file_path: Path) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loaders.py
import hashlib
from pathlib import Path
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader


def test_compute_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    loader = DocumentLoader(Settings())
    h = loader.compute_hash(f)
    expected = hashlib.md5(b"hello world").hexdigest()
    assert h == expected


def test_load_markdown_file(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Title\n\nBody text here.")
    loader = DocumentLoader(Settings())
    docs = loader.load_files([f])
    assert len(docs) > 0
    assert "Title" in docs[0].text or "Body" in docs[0].text


def test_load_text_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Plain text content.")
    loader = DocumentLoader(Settings())
    docs = loader.load_files([f])
    assert len(docs) > 0


def test_load_nonexistent_file(tmp_path):
    loader = DocumentLoader(Settings())
    docs = loader.load_files([tmp_path / "nonexistent.pdf"])
    assert len(docs) == 0  # Failed files are skipped, not fatal


def test_large_file_warning(tmp_path, caplog):
    settings = Settings(WARN_FILE_SIZE_MB=0)  # 0 MB = warn on everything
    f = tmp_path / "large.txt"
    f.write_text("x" * 100)
    loader = DocumentLoader(settings)
    docs = loader.load_files([f])
    assert len(docs) > 0  # Still loads, just warns


def test_file_too_large_rejected(tmp_path):
    settings = Settings(MAX_FILE_SIZE_MB=0)  # 0 MB = reject everything
    f = tmp_path / "huge.txt"
    f.write_text("x")
    loader = DocumentLoader(settings)
    docs = loader.load_files([f])
    assert len(docs) == 0  # Rejected
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_loaders.py::test_compute_hash -v`
Expected: ImportError.

- [ ] **Step 3: Implement DocumentLoader**

```python
import hashlib
import structlog
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from knowledge_hub.config import Settings

logger = structlog.get_logger()

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm", ".docx", ".rst"}


class DocumentLoader:
    """Loads documents from files, dispatching by format.

    Uses LlamaIndex SimpleDirectoryReader for format detection.
    PDFs are converted to markdown for heading preservation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def load_files(self, paths: list[Path]) -> list[Document]:
        valid_paths = []
        for p in paths:
            if not p.exists():
                logger.warning("file_not_found", path=str(p))
                continue
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > self.settings.MAX_FILE_SIZE_MB:
                logger.warning("file_too_large_rejected", path=str(p), size_mb=size_mb)
                continue
            if size_mb > self.settings.WARN_FILE_SIZE_MB:
                logger.warning("large_file_warning", path=str(p), size_mb=size_mb)
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                logger.warning("unsupported_format", path=str(p), suffix=p.suffix)
                continue
            valid_paths.append(p)

        if not valid_paths:
            return []

        reader = SimpleDirectoryReader(
            input_files=[str(p) for p in valid_paths],
            recursive=False,
        )
        documents = reader.load_data()
        logger.info("files_loaded", count=len(documents), files=len(valid_paths))
        return documents

    def compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for incremental update detection."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_loaders.py -v`
Expected: all PASS (text and markdown tests pass; PDF/DOCX tests may skip depending on dependencies).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/ingestion/loaders.py tests/test_loaders.py
git commit -m "feat: add DocumentLoader with format dispatch, size limits, and hash computation"
```

---

### Task 8: Ingestion — Pipeline (Orchestrator)

**Files:**
- Create: `src/knowledge_hub/ingestion/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Settings`, `DocumentLoader`, `SemanticChunker`, `OllamaEmbedder`, `QdrantVectorStore`, `SourceMetadataManager`
- Produces: `IngestionPipeline` class
  - `async run(paths: list[Path] | None = None, force: bool = False, tags: list[str] = []) -> IngestionReport`

- [ ] **Step 1: Write failing test for pipeline**

```python
# tests/test_pipeline.py
import hashlib
from pathlib import Path
import pytest
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def settings(temp_storage_dir, tmp_path):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_pipeline",
        STORAGE_DIR=str(temp_storage_dir),
        DATA_DIR=str(tmp_path / "data"),
    )


@pytest.fixture
async def pipeline(settings):
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    loader = DocumentLoader(settings)
    chunker = SemanticChunker(settings)
    embedder = OllamaEmbedder(settings)
    pl = IngestionPipeline(settings, loader, chunker, embedder, store, meta_mgr)
    yield pl
    client.delete_collection(settings.QDRANT_COLLECTION)
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.asyncio
async def test_pipeline_ingests_markdown(pipeline, settings, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.md").write_text("# Hello\n\nWorld content here. " * 10)

    report = await pipeline.run([data_dir / "test.md"])

    assert report.succeeded == 1
    assert report.failed == 0
    # Re-run should skip (same hash)
    report2 = await pipeline.run([data_dir / "test.md"])
    assert report2.skipped == 1


@pytest.mark.asyncio
async def test_pipeline_handles_missing_file(pipeline, settings, tmp_path):
    report = await pipeline.run([tmp_path / "nonexistent.md"])
    assert report.succeeded == 0
    assert report.failed == 0  # Missing files are logged, not counted as failures
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_pipeline.py::test_pipeline_ingests_markdown -v`
Expected: ImportError.

- [ ] **Step 3: Implement IngestionPipeline**

```python
import hashlib
import json
import structlog
from pathlib import Path
from dataclasses import dataclass, field
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager

logger = structlog.get_logger()


@dataclass
class IngestionReport:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    orphans_cleaned: int = 0
    failed_files: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrates the full ingestion flow: load → chunk → embed → store.

    Handles incremental updates via source_hash comparison and
    orphan vector cleanup after ingestion.
    """

    def __init__(
        self,
        settings: Settings,
        loader: DocumentLoader,
        chunker: SemanticChunker,
        embedder: OllamaEmbedder,
        vector_store: QdrantVectorStore,
        metadata_mgr: SourceMetadataManager,
    ):
        self.settings = settings
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = vector_store
        self._metadata = metadata_mgr

    async def run(
        self,
        paths: list[Path] | None = None,
        force: bool = False,
        tags: list[str] | None = None,
    ) -> IngestionReport:
        tags = tags or []
        report = IngestionReport()

        if paths is None:
            data_dir = Path(self.settings.DATA_DIR)
            paths = list(data_dir.rglob("*")) if data_dir.exists() else []

        files = [p for p in paths if p.is_file()]
        report.total = len(files)
        if not files:
            logger.info("no_files_to_ingest")
            return report

        for file_path in files:
            try:
                source_hash = self._loader.compute_hash(file_path)
                source_name = file_path.name

                # Check for existing hash (incremental update)
                if not force:
                    existing_hash = await self._metadata.get_hash(source_name)
                    if existing_hash == source_hash:
                        logger.debug("file_unchanged_skipped", file=source_name)
                        report.skipped += 1
                        continue

                # Remove old chunks if re-ingesting changed file
                if existing_hash := await self._metadata.get_hash(source_name):
                    await self._store.delete_by_source(source_name)

                # Load sidecar metadata
                file_tags = list(tags)
                sidecar = file_path.parent / ".meta.json"
                if sidecar.exists():
                    sidecar_data = json.loads(sidecar.read_text())
                    if "tags" in sidecar_data:
                        file_tags = sidecar_data["tags"]  # sidecar overrides CLI tags
                # Directory name as fallback tag
                dir_tag = file_path.parent.name
                if dir_tag and dir_tag not in file_tags:
                    file_tags.append(dir_tag)

                # Load → Chunk → Embed → Store
                docs = self._loader.load_files([file_path])
                if not docs:
                    continue

                chunks = self._chunker.chunk(docs, source_name, source_hash)
                if not chunks:
                    continue

                # Embed all chunks
                texts = [c.text for c in chunks]
                embeddings = await self._embedder.embed_texts(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.dense_embedding = emb["dense"]
                    chunk.sparse_embedding = emb["sparse"]
                    # Apply tags
                    chunk.metadata.tags = file_tags

                # Store
                await self._store.upsert_chunks(chunks)
                await self._metadata.upsert(source_name, source_hash, len(chunks))

                report.succeeded += 1
                logger.info("file_ingested", file=source_name, chunks=len(chunks))

            except Exception as e:
                logger.error("ingestion_failed", file=str(file_path), error=str(e))
                report.failed += 1
                report.failed_files.append(str(file_path))

        # Orphan cleanup: remove vectors for files no longer on disk
        local_files = {p.name for p in paths if p.is_file()}
        report.orphans_cleaned = await self._metadata.orphan_cleanup(local_files)

        logger.info(
            "ingestion_complete",
            total=report.total,
            succeeded=report.succeeded,
            failed=report.failed,
            skipped=report.skipped,
            orphans=report.orphans_cleaned,
        )
        return report
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: tests pass (requires Qdrant and Ollama running).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: add IngestionPipeline with incremental updates, sidecar tags, orphan cleanup"
```

---

### Task 9: Retrieval — Reranker

**Files:**
- Create: `src/knowledge_hub/retrieval/reranker.py`
- Create: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `Settings` from `config.py`
- Produces: `Reranker` class
  - `async rerank(query: str, candidates: list[dict]) -> list[dict]` — reorders and scores candidates; on failure returns unmodified list

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reranker.py
import pytest
from knowledge_hub.config import Settings
from knowledge_hub.retrieval.reranker import Reranker


@pytest.fixture
def settings():
    return Settings(OLLAMA_BASE_URL="http://localhost:11434", RERANK_MODEL="bge-reranker")


@pytest.fixture
def reranker(settings):
    return Reranker(settings)


@pytest.mark.asyncio
async def test_rerank_returns_top_k(reranker):
    candidates = [
        {"text": "The quick brown fox jumps over the lazy dog", "score": 0.8},
        {"text": "Priority inheritance prevents priority inversion", "score": 0.7},
        {"text": "Foxes are omnivorous mammals", "score": 0.6},
    ]
    query = "What is priority inheritance?"
    results = await reranker.rerank(query, candidates, top_k=2)
    assert len(results) == 2
    # The RTOS-related text should rank higher after reranking
    assert "priority" in results[0]["text"].lower()


@pytest.mark.asyncio
async def test_rerank_graceful_degradation(settings):
    """When Ollama is unreachable, reranker should return candidates unchanged."""
    settings.OLLAMA_BASE_URL = "http://nonexistent-host:99999"
    broken_reranker = Reranker(settings)
    candidates = [{"text": "test", "score": 0.5}]
    results = await broken_reranker.rerank("query", candidates, top_k=1)
    assert results == candidates  # Unmodified — graceful degradation
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_reranker.py::test_rerank_graceful_degradation -v`
Expected: ImportError.

- [ ] **Step 3: Implement Reranker**

```python
import structlog
import httpx
from knowledge_hub.config import Settings

logger = structlog.get_logger()


class Reranker:
    """Cross-encoder reranker using bge-reranker via Ollama.

    On failure: graceful degradation — returns candidates in original order.
    """

    def __init__(self, settings: Settings):
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.RERANK_MODEL

    async def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5
    ) -> list[dict]:
        if not candidates:
            return candidates

        try:
            scored = await self._call_reranker(query, candidates)
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning("reranker_failed_degrading", error=str(e))
            # Graceful degradation: return candidates as-is, capped to top_k
            return candidates[:top_k]

    async def _call_reranker(self, query: str, candidates: list[dict]) -> list[dict]:
        async with httpx.AsyncClient(timeout=60) as client:
            results = []
            for c in candidates:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": f"Query: {query}\nDocument: {c['text']}\nRelevance score (0-1):",
                        "stream": False,
                    },
                )
                if response.status_code == 200:
                    try:
                        score = float(response.json()["response"].strip())
                        results.append({**c, "score": max(0.0, min(1.0, score))})
                    except (ValueError, KeyError):
                        results.append(c)
                else:
                    results.append(c)
            return results
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_reranker.py -v`
Expected: `test_rerank_graceful_degradation` PASSES; `test_rerank_returns_top_k` requires Ollama with bge-reranker.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/retrieval/reranker.py tests/test_reranker.py
git commit -m "feat: add Reranker with bge-reranker and graceful degradation on failure"
```

---

### Task 10: Retrieval — QueryEngine

**Files:**
- Create: `src/knowledge_hub/retrieval/query_engine.py`
- Create: `tests/test_query_engine.py`

**Interfaces:**
- Consumes: `Settings`, `OllamaEmbedder`, `QdrantVectorStore`, `Reranker`, `QueryInput`, `QueryResult`, `ChunkResult` from schemas
- Produces: `QueryEngine` class
  - `async query(q: QueryInput) -> QueryResult`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_query_engine.py
import pytest
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput, DocumentChunk, ChunkMetadata


@pytest.fixture
def settings(temp_storage_dir):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_query_eng",
        OLLAMA_BASE_URL="http://localhost:11434",
        STORAGE_DIR=str(temp_storage_dir),
    )


@pytest.fixture
async def query_engine(settings):
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    embedder = OllamaEmbedder(settings)
    reranker = Reranker(settings)
    engine = QueryEngine(settings, embedder, store, reranker)
    yield engine
    client.delete_collection(settings.QDRANT_COLLECTION)
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")


@pytest.mark.asyncio
async def test_query_empty_collection(query_engine):
    result = await query_engine.query(QueryInput(query="test"))
    assert len(result.results) == 0
    assert result.query_time_ms >= 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_query_engine.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement QueryEngine**

```python
import time
import structlog
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.schemas import QueryInput, QueryResult, ChunkResult

logger = structlog.get_logger()


class QueryEngine:
    """Orchestrates the full query flow: embed → hybrid search → rerank.

    Depends on OllamaEmbedder for query embedding, QdrantVectorStore for
    hybrid search, and Reranker for cross-encoder reranking.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: OllamaEmbedder,
        vector_store: QdrantVectorStore,
        reranker: Reranker,
    ):
        self.settings = settings
        self._embedder = embedder
        self._store = vector_store
        self._reranker = reranker

    async def query(self, q: QueryInput) -> QueryResult:
        start = time.perf_counter()

        # 1. Embed the query
        query_embedding = await self._embedder.embed_query(q.query)

        # 2. Hybrid search
        candidates = await self._store.hybrid_search(
            dense_vec=query_embedding["dense"],
            sparse_vec=query_embedding["sparse"],
            top_k=self.settings.HYBRID_CANDIDATE_K,
            filter_source=q.filter_source,
            filter_tags=q.filter_tags,
        )

        if not candidates:
            elapsed = (time.perf_counter() - start) * 1000
            return QueryResult(results=[], query_time_ms=elapsed)

        # 3. Rerank
        candidate_dicts = [
            {"text": payload.get("text", ""), "score": score, "payload": payload}
            for (_id, score, payload) in candidates
        ]
        reranked = await self._reranker.rerank(q.query, candidate_dicts, top_k=q.top_k)

        # 4. Build result
        results = [
            ChunkResult(
                text=c["text"],
                source_file=c["payload"].get("source_file", "unknown"),
                page_or_section=f"p{c['payload'].get('page_number', '?')}",
                heading_path=c["payload"].get("heading_path", []),
                score=c["score"],
            )
            for c in reranked
        ]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("query_complete", query=q.query[:80], results=len(results), time_ms=elapsed)
        return QueryResult(results=results, query_time_ms=elapsed)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_query_engine.py -v`
Expected: `test_query_empty_collection` PASSES (requires Qdrant).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/retrieval/query_engine.py tests/test_query_engine.py
git commit -m "feat: add QueryEngine with embed→hybrid search→rerank pipeline"
```

---

### Task 11: Server — Health Monitor + MCP Tools + MCP Server

**Files:**
- Create: `src/knowledge_hub/server/health.py`
- Create: `src/knowledge_hub/server/tools.py`
- Create: `src/knowledge_hub/server/mcp_server.py`

**Interfaces:**
- Consumes: `Settings`, `QueryEngine`, `HealthMonitor`, `QueryInput`, `QueryResult` from schemas
- Produces: `FastMCP` application instance with `query_knowledge_base` tool and auth

- [ ] **Step 1: Implement HealthMonitor**

```python
# src/knowledge_hub/server/health.py
import asyncio
import structlog
from dataclasses import dataclass
import httpx
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings

logger = structlog.get_logger()


@dataclass
class HealthStatus:
    ollama: bool = False
    qdrant: bool = False
    gpu_available: bool = False
    gpu_memory_free_mb: int = 0


class HealthMonitor:
    """Background health prober for Ollama, Qdrant, and GPU.

    Caches status and refreshes on a configurable interval.
    QueryEngine checks get_status() before making calls.
    """

    def __init__(self, settings: Settings, qdrant_client: QdrantClient):
        self.settings = settings
        self._qdrant = qdrant_client
        self._ollama_url = settings.OLLAMA_BASE_URL
        self._cached_status: HealthStatus | None = None
        self._task: asyncio.Task | None = None

    async def start(self, interval_seconds: int = 30):
        self._cached_status = await self._probe_all()
        self._task = asyncio.create_task(self._probe_loop(interval_seconds))

    async def get_status(self) -> HealthStatus:
        if self._cached_status is None:
            self._cached_status = await self._probe_all()
        return self._cached_status

    async def _probe_loop(self, interval: int):
        while True:
            try:
                self._cached_status = await self._probe_all()
            except Exception as e:
                logger.error("health_probe_failed", error=str(e))
            await asyncio.sleep(interval)

    async def _probe_all(self) -> HealthStatus:
        status = HealthStatus()
        status.ollama = await self._probe_ollama()
        status.qdrant = await self._probe_qdrant()
        status.gpu_available, status.gpu_memory_free_mb = await self._probe_gpu()
        return status

    async def _probe_ollama(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def _probe_qdrant(self) -> bool:
        try:
            collections = self._qdrant.get_collections()
            return collections is not None
        except Exception:
            return False

    async def _probe_gpu(self) -> tuple[bool, int]:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_free = int(result.stdout.strip().split("\n")[0])
                return True, mem_free
        except Exception:
            pass
        return False, 0
```

- [ ] **Step 2: Implement MCP Tools**

```python
# src/knowledge_hub/server/tools.py
import structlog
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput
from knowledge_hub.config import Settings

logger = structlog.get_logger()


def create_tools(settings: Settings, query_engine: QueryEngine, health: HealthMonitor):
    """Create MCP tool wrappers around the QueryEngine."""

    async def query_knowledge_base(
        query: str,
        top_k: int = 5,
        filter_source: str | None = None,
        filter_tags: list[str] | None = None,
    ) -> dict:
        """Search the knowledge base for relevant document chunks.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default 5).
            filter_source: Optional filter by source filename.
            filter_tags: Optional filter by document tags.

        Returns:
            Dictionary with 'results' list and 'query_time_ms' float.
        """
        # Check health first
        status = await health.get_status()
        if not status.ollama:
            return {"error": "Ollama is not available", "results": [], "query_time_ms": 0}
        if not status.qdrant:
            return {"error": "Knowledge base is not available", "results": [], "query_time_ms": 0}

        q_input = QueryInput(
            query=query,
            top_k=top_k,
            filter_source=filter_source,
            filter_tags=filter_tags,
        )
        result = await query_engine.query(q_input)
        return result.model_dump()

    return {"query_knowledge_base": query_knowledge_base}
```

- [ ] **Step 3: Implement MCP Server**

```python
# src/knowledge_hub/server/mcp_server.py
import structlog
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.server.health import HealthMonitor
from knowledge_hub.server.tools import create_tools

logger = structlog.get_logger()


def create_mcp_app(settings: Settings) -> FastMCP:
    """Build and configure the FastMCP application.

    Wires together all components: health monitor, embedder, vector store,
    reranker, query engine, and MCP tools.
    """
    # Infrastructure clients
    qdrant_client = QdrantClient(settings.QDRANT_URL)

    # Health monitor
    health = HealthMonitor(settings, qdrant_client)

    # Storage
    metadata_mgr = SourceMetadataManager(settings, qdrant_client)
    vector_store = QdrantVectorStore(settings, qdrant_client, metadata_mgr)

    # Retrieval components
    embedder = OllamaEmbedder(settings)
    reranker = Reranker(settings)
    query_engine = QueryEngine(settings, embedder, vector_store, reranker)

    # Auth
    if settings.MCP_AUTH_TOKEN:
        from fastmcp.server.auth import StaticTokenVerifier
        verifier = StaticTokenVerifier(settings.MCP_AUTH_TOKEN)
        mcp = FastMCP("knowledge-hub", auth=verifier)
    else:
        if settings.MCP_HOST != "127.0.0.1":
            raise ValueError(
                "MCP_HOST must be 127.0.0.1 when MCP_AUTH_TOKEN is not set. "
                "Set KH_MCP_AUTH_TOKEN to enable LAN access."
            )
        mcp = FastMCP("knowledge-hub")

    # IP allowlist middleware
    if settings.MCP_ALLOWED_IPS:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class IPAllowlistMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                client_ip = request.client.host if request.client else None
                if client_ip not in settings.MCP_ALLOWED_IPS:
                    return JSONResponse({"error": "Forbidden"}, status_code=403)
                return await call_next(request)
        mcp.add_middleware(IPAllowlistMiddleware)

    # Register tools
    tools = create_tools(settings, query_engine, health)
    for name, fn in tools.items():
        mcp.add_tool(fn, name=name)

    # Store health monitor for startup
    mcp._health = health  # accessed by CLI start command

    return mcp


async def run_mcp_server(settings: Settings):
    """Start the MCP server with health monitoring."""
    mcp = create_mcp_app(settings)

    # Start health probe loop
    await mcp._health.start()

    transport = "sse" if settings.MCP_TRANSPORT == "sse" else "streamable-http"
    logger.info(
        "mcp_server_starting",
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        transport=transport,
        auth=bool(settings.MCP_AUTH_TOKEN),
    )

    await mcp.run(
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        transport=transport,
    )
```

- [ ] **Step 4: Run verification**

```bash
uv run python -c "from knowledge_hub.server.mcp_server import create_mcp_app; from knowledge_hub.config import Settings; mcp = create_mcp_app(Settings()); print('MCP app created successfully')"
```

Expected: prints "MCP app created successfully".

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_hub/server/
git commit -m "feat: add HealthMonitor, MCP tools, and FastMCP server with auth and IP filtering"
```

---

### Task 12: CLI

**Files:**
- Create: `src/knowledge_hub/cli/main.py`

**Interfaces:**
- Consumes: `Settings`, `IngestionPipeline`, `QueryEngine`, `run_mcp_server`
- Produces: Click CLI with commands: `index`, `query`, `status`, `cleanup-orphans`, `config`, `serve`

- [ ] **Step 1: Implement CLI**

```python
"""CLI for knowledge-hub: index, query, manage, and serve."""
import asyncio
from pathlib import Path
import structlog
import click
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.ingestion.pipeline import IngestionPipeline
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput

logger = structlog.get_logger()


def _get_settings() -> Settings:
    return Settings()


def _build_pipeline(settings: Settings) -> IngestionPipeline:
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    store = QdrantVectorStore(settings, client, meta_mgr)
    return IngestionPipeline(
        settings=settings,
        loader=DocumentLoader(settings),
        chunker=SemanticChunker(settings),
        embedder=OllamaEmbedder(settings),
        vector_store=store,
        metadata_mgr=meta_mgr,
    )


def _build_query_engine(settings: Settings) -> QueryEngine:
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    store = QdrantVectorStore(settings, client, meta_mgr)
    embedder = OllamaEmbedder(settings)
    reranker = Reranker(settings)
    return QueryEngine(settings, embedder, store, reranker)


@click.group()
def cli():
    """knowledge-hub — Local Vector RAG knowledge base."""
    pass


@cli.command()
@click.option("--path", type=click.Path(exists=True), default=None, help="Directory to ingest from.")
@click.option("--force", is_flag=True, help="Re-ingest all files, ignoring source hash cache.")
@click.option("--tags", default=None, help="Comma-separated tags (overridden by .meta.json sidecars).")
def index(path, force, tags):
    """Ingest documents into the knowledge base."""
    settings = _get_settings()
    pipeline = _build_pipeline(settings)
    paths = [Path(path)] if path else None
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    report = asyncio.run(pipeline.run(paths=paths, force=force, tags=tag_list))
    click.echo(f"Total: {report.total}, Succeeded: {report.succeeded}, "
               f"Failed: {report.failed}, Skipped: {report.skipped}, "
               f"Orphans cleaned: {report.orphans_cleaned}")
    if report.failed_files:
        click.echo("Failed files:")
        for f in report.failed_files:
            click.echo(f"  - {f}")


@cli.command()
@click.argument("query_text")
@click.option("-k", "--top-k", type=int, default=5, help="Number of results.")
def query(query_text, top_k):
    """Query the knowledge base directly."""
    settings = _get_settings()
    engine = _build_query_engine(settings)
    result = asyncio.run(engine.query(QueryInput(query=query_text, top_k=top_k)))
    click.echo(f"Results ({result.query_time_ms:.1f}ms):")
    for i, r in enumerate(result.results):
        heading = " > ".join(r.heading_path) if r.heading_path else "(no heading)"
        click.echo(f"\n--- Result {i+1} (score: {r.score:.3f}) ---")
        click.echo(f"Source: {r.source_file} | {r.page_or_section} | {heading}")
        click.echo(r.text[:500])


@cli.command()
def status():
    """Show knowledge base status."""
    settings = _get_settings()
    client = QdrantClient(settings.QDRANT_URL)
    try:
        count = client.count(collection_name=settings.QDRANT_COLLECTION).count
        meta_mgr = SourceMetadataManager(settings, client)
        sources = asyncio.run(meta_mgr.list_sources())
        click.echo(f"Collection: {settings.QDRANT_COLLECTION}")
        click.echo(f"Total chunks: {count}")
        click.echo(f"Source files: {len(sources)}")
        if sources:
            click.echo("Sources:")
            for s in sorted(sources)[:20]:
                click.echo(f"  - {s}")
            if len(sources) > 20:
                click.echo(f"  ... and {len(sources) - 20} more")
    except Exception as e:
        click.echo(f"Error connecting to Qdrant: {e}")


@cli.command()
def cleanup_orphans():
    """Remove vectors for deleted source files."""
    settings = _get_settings()
    data_dir = Path(settings.DATA_DIR)
    local_files = {p.name for p in data_dir.rglob("*") if p.is_file()} if data_dir.exists() else set()
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    removed = asyncio.run(meta_mgr.orphan_cleanup(local_files))
    click.echo(f"Removed {removed} orphan source(s).")


@cli.group()
def config():
    """Manage configuration."""
    pass


@config.command("show")
def config_show():
    """Show current effective configuration."""
    settings = _get_settings()
    for field, value in settings.model_dump().items():
        # Mask sensitive values
        if "token" in field.lower() and value:
            value = value[:4] + "****"
        click.echo(f"KH_{field}={value}")


@config.command("reset-batch-size")
def config_reset_batch_size():
    """Reset OOM-degraded batch size to default."""
    settings = _get_settings()
    embedder = OllamaEmbedder(settings)
    asyncio.run(embedder.reset_batch_size())
    click.echo(f"Batch size reset to {settings.EMBED_BATCH_SIZE}")


@cli.command()
@click.option("--host", default=None, help="Bind address.")
@click.option("--port", default=None, type=int, help="Bind port.")
def serve(host, port):
    """Start the MCP server."""
    settings = _get_settings()
    if host:
        settings.MCP_HOST = host
    if port:
        settings.MCP_PORT = port
    from knowledge_hub.server.mcp_server import run_mcp_server
    asyncio.run(run_mcp_server(settings))


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Verify CLI loads**

Run: `kh --help`
Expected: shows command list with `index`, `query`, `status`, `cleanup-orphans`, `config`, `serve`.

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_hub/cli/main.py
git commit -m "feat: add CLI with index, query, status, cleanup-orphans, config, serve commands"
```

---

### Task 13: Integration Tests

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: ingest a markdown file and query it."""
import pytest
from pathlib import Path
from qdrant_client import QdrantClient
from knowledge_hub.config import Settings
from knowledge_hub.ingestion.loaders import DocumentLoader
from knowledge_hub.ingestion.chunker import SemanticChunker
from knowledge_hub.ingestion.embedder import OllamaEmbedder
from knowledge_hub.ingestion.pipeline import IngestionPipeline
from knowledge_hub.storage.vector_store import QdrantVectorStore
from knowledge_hub.storage.metadata import SourceMetadataManager
from knowledge_hub.retrieval.reranker import Reranker
from knowledge_hub.retrieval.query_engine import QueryEngine
from knowledge_hub.schemas import QueryInput


@pytest.fixture
def settings(temp_storage_dir, tmp_path):
    return Settings(
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_integration",
        OLLAMA_BASE_URL="http://localhost:11434",
        STORAGE_DIR=str(temp_storage_dir),
        DATA_DIR=str(tmp_path / "data"),
    )


@pytest.mark.asyncio
async def test_full_ingest_and_query(settings, tmp_path):
    """Ingest a markdown document and verify it can be queried."""
    # Setup test document
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc_path = data_dir / "rtos_guide.md"
    doc_path.write_text("""# FreeRTOS Scheduling Guide

## Priority Inheritance

Priority inheritance is a mechanism that prevents priority inversion.
When a high-priority task is blocked by a low-priority task holding a mutex,
the low-priority task temporarily inherits the high priority until it
releases the mutex. This ensures the high-priority task doesn't get
indefinitely blocked by medium-priority tasks.

## Task Notifications

Task notifications are a lightweight alternative to semaphores in FreeRTOS.
Each task has a 32-bit notification value that can be updated by other tasks
or interrupts. Notifications are faster and use less RAM than semaphores.
""")

    # Build components
    client = QdrantClient(settings.QDRANT_URL)
    meta_mgr = SourceMetadataManager(settings, client)
    await meta_mgr.ensure_collection()
    store = QdrantVectorStore(settings, client, meta_mgr)
    await store.ensure_collection()
    embedder = OllamaEmbedder(settings)

    pipeline = IngestionPipeline(
        settings=settings,
        loader=DocumentLoader(settings),
        chunker=SemanticChunker(settings),
        embedder=embedder,
        vector_store=store,
        metadata_mgr=meta_mgr,
    )

    # Ingest
    report = await pipeline.run([doc_path], tags=["rtos", "freertos"])
    assert report.succeeded == 1

    # Query
    reranker = Reranker(settings)
    engine = QueryEngine(settings, embedder, store, reranker)
    result = await engine.query(QueryInput(query="priority inheritance mutex", top_k=2))

    assert len(result.results) > 0
    assert result.query_time_ms > 0
    # The result should contain relevant content
    found = any("inheritance" in r.text.lower() for r in result.results)
    assert found, f"Expected 'inheritance' in results, got: {[r.text[:100] for r in result.results]}"

    # Cleanup
    client.delete_collection(settings.QDRANT_COLLECTION)
    client.delete_collection(f"{settings.QDRANT_COLLECTION}_source_meta")
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v -s`
Expected: test passes (requires Qdrant + Ollama with bge-m3 and bge-reranker).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for ingest→query flow"
```

---

### Task 14: Final Verification & Graphify Update

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --no-header`
Expected: all unit tests pass; integration tests pass or skip depending on local services.

- [ ] **Step 2: Verify CLI end-to-end**

```bash
mkdir -p data
echo "# Test\n\nThis is a test document about Python async programming." > data/test.md
kh index
kh query "Python async" -k 3
kh status
```

Expected: ingestion succeeds, query returns results, status shows collection info.

- [ ] **Step 3: Update graphify**

Run: `graphify update .`

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "chore: final verification, graphify update, and cleanup"
```

---

