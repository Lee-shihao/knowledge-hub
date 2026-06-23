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