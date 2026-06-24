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
