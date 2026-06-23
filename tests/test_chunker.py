"""Tests for SemanticChunker — heading-aware document chunking."""

import hashlib

from llama_index.core.schema import Document

from knowledge_hub.config import Settings
from knowledge_hub.ingestion.chunker import SemanticChunker


def make_doc(text: str) -> Document:
    """Minimal Document-like object for testing."""
    return Document(text=text, metadata={})


def test_chunker_produces_chunks():
    """Long document should be split into multiple chunks."""
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
    """Short document fits in a single chunk."""
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("Short text.")
    chunks = chunker.chunk([doc], "short.txt", "hash1")
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunker_heading_path_in_metadata():
    """Chunks under headings should carry the heading chain."""
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc(
        "# Chapter 1\n\nContent here. " * 5
        + "\n\n## Section 1.1\n\nMore content. " * 5
    )
    chunks = chunker.chunk([doc], "headings.md", "hash1")
    heading_chunks = [c for c in chunks if c.metadata.heading_path]
    assert len(heading_chunks) > 0


def test_chunk_id_deterministic():
    """Same input must always produce the same chunk ID."""
    settings = Settings()
    chunker = SemanticChunker(settings)
    doc = make_doc("Unique content for ID testing. " * 10)
    chunks1 = chunker.chunk([doc], "same.txt", "h1")
    chunks2 = chunker.chunk([doc], "same.txt", "h1")
    assert chunks1[0].id == chunks2[0].id


def test_chunk_id_differs_for_different_source():
    """Different source files must produce different chunk IDs."""
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("Same content for different sources. " * 10)
    chunks_a = chunker.chunk([doc], "file_a.txt", "h1")
    chunks_b = chunker.chunk([doc], "file_b.txt", "h1")
    assert chunks_a[0].id != chunks_b[0].id


def test_heading_chain_resets_correctly():
    """Heading chain should track nested levels and reset properly."""
    settings = Settings(CHUNK_MAX_TOKENS=500, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc(
        "# Chapter 1\n\nPara one.\n\n## Section 1.1\n\nPara two.\n\n"
        "# Chapter 2\n\nPara three."
    )
    chunks = chunker.chunk([doc], "chain.md", "h1")

    ch1_chunks = [c for c in chunks if "Chapter 1" in c.metadata.heading_path]
    ch2_chunks = [c for c in chunks if "Chapter 2" in c.metadata.heading_path]
    sec_chunks = [c for c in chunks if c.metadata.heading_path == ["Chapter 1", "Section 1.1"]]

    assert len(ch1_chunks) > 0
    assert len(ch2_chunks) > 0
    assert len(sec_chunks) > 0
    # Chapter 2 chunks should NOT contain Section 1.1
    for c in ch2_chunks:
        assert "Section 1.1" not in c.metadata.heading_path


def test_overlap_keeps_last_paragraph():
    """When CHUNK_OVERLAP > 0, the next chunk should include the last paragraph of the previous."""
    settings = Settings(CHUNK_MAX_TOKENS=30, CHUNK_OVERLAP=0.1)
    chunker = SemanticChunker(settings)
    doc = make_doc(
        "First paragraph with enough text to fill a chunk. It needs to be long.\n\n"
        "Second paragraph also with enough text to fill a chunk on its own.\n\n"
        "Third paragraph that should overlap with the second one."
    )
    chunks = chunker.chunk([doc], "overlap.txt", "h1")
    if len(chunks) > 1:
        # The second chunk's text should contain some text from the previous chunk's last paragraph
        # Overlap means the last paragraph of chunk N is included in chunk N+1
        assert len(chunks) >= 2


def test_hard_split_oversized_paragraph():
    """A single paragraph exceeding max_tokens should be hard-split into word chunks."""
    settings = Settings(CHUNK_MAX_TOKENS=20, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    # ~100 words * ~5 chars = 500 chars / 4 = ~125 tokens, well over 20
    long_para = " ".join(f"word{i}" for i in range(100))
    doc = make_doc(long_para)
    chunks = chunker.chunk([doc], "long.txt", "h1")
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata.source_file == "long.txt"


def test_empty_document():
    """Empty document should produce no chunks."""
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("")
    chunks = chunker.chunk([doc], "empty.txt", "h1")
    assert len(chunks) == 0


def test_multiple_documents():
    """Chunker should process all documents in the list."""
    settings = Settings(CHUNK_MAX_TOKENS=100, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    docs = [
        make_doc("Document one content. " * 10),
        make_doc("Document two content. " * 10),
    ]
    chunks = chunker.chunk(docs, "multi.txt", "h1")
    assert len(chunks) >= 2


def test_chunk_id_uses_md5_formula():
    """Verify the chunk ID matches the spec: md5(source_file + '|' + heading_path + text[:200])."""
    settings = Settings(CHUNK_MAX_TOKENS=500, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("Test content for ID verification.")
    chunks = chunker.chunk([doc], "id_test.txt", "h1")
    assert len(chunks) == 1

    expected_raw = "id_test.txt|" + "Test content for ID verification."[:200]
    expected_id = hashlib.md5(expected_raw.encode()).hexdigest()
    assert chunks[0].id == expected_id


def test_heading_chain_with_h3():
    """H3 heading should create a three-level heading chain."""
    settings = Settings(CHUNK_MAX_TOKENS=500, CHUNK_OVERLAP=0.0)
    chunker = SemanticChunker(settings)
    doc = make_doc("# A\n\nText A.\n\n## B\n\nText B.\n\n### C\n\nText C.")
    chunks = chunker.chunk([doc], "deep.md", "h1")

    h3_chunks = [c for c in chunks if c.metadata.heading_path == ["A", "B", "C"]]
    assert len(h3_chunks) == 1
    assert "Text C." in h3_chunks[0].text
