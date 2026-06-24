from datetime import UTC, datetime
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source_file: str
    source_hash: str
    page_number: int | None = None
    heading_path: list[str] = []
    tags: list[str] = []
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
