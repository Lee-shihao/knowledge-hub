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


def test_unsupported_suffix_skipped(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2")
    loader = DocumentLoader(Settings())
    docs = loader.load_files([f])
    assert len(docs) == 0


def test_supported_suffixes():
    from knowledge_hub.ingestion.loaders import SUPPORTED_SUFFIXES

    expected = {".pdf", ".md", ".txt", ".html", ".htm", ".docx", ".rst"}
    assert SUPPORTED_SUFFIXES == expected
