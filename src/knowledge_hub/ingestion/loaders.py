import hashlib
import re

import structlog
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from knowledge_hub.config import Settings

logger = structlog.get_logger()

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm", ".docx", ".rst"}

# PDF text extraction can produce raw glyph names (e.g. /uni00000019) when
# pypdf can't decode a font's CMap. This pattern detects those garbage pages.
_GLYPH_GARBAGE_RE = re.compile(r"/uni[0-9a-f]{4,}")


class DocumentLoader:
    """Loads documents from files, dispatching by format.

    PDFs are extracted with pypdf using garbage-detection fallback:
    when a page's text is dominated by raw glyph names (e.g. /uni00000019),
    the page is re-extracted with extraction_mode="layout" to recover
    readable text from custom-encoded fonts.

    Other formats use LlamaIndex SimpleDirectoryReader.
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

        # Separate PDF and non-PDF: PDFs get custom extraction with
        # glyph-garbage detection + layout-mode fallback
        pdf_paths = [p for p in valid_paths if p.suffix.lower() == ".pdf"]
        other_paths = [p for p in valid_paths if p.suffix.lower() != ".pdf"]

        documents: list[Document] = []

        for pdf_path in pdf_paths:
            pdf_docs = self._load_pdf(pdf_path)
            documents.extend(pdf_docs)

        if other_paths:
            reader = SimpleDirectoryReader(
                input_files=[str(p) for p in other_paths],
                recursive=False,
            )
            documents.extend(reader.load_data())

        logger.info(
            "files_loaded",
            pages=len(documents),
            files=len(valid_paths),
        )
        return documents

    @staticmethod
    def _load_pdf(file_path: Path) -> list[Document]:
        """Load a single PDF, one Document per page, with garbage detection."""
        from pypdf import PdfReader

        documents: list[Document] = []
        reader = PdfReader(str(file_path))

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Detect pages where default extraction produced raw glyph names
            # (broken CMap / custom font encoding). Re-extract with layout mode.
            garbage_matches = len(_GLYPH_GARBAGE_RE.findall(text))
            if garbage_matches > 50:
                logger.debug(
                    "pdf_glyph_garbage_detected",
                    file=file_path.name,
                    page=page_idx,
                    garbage_tokens=garbage_matches,
                )
                cleaned = page.extract_text(extraction_mode="layout")
                if cleaned and cleaned.strip():
                    text = cleaned

            documents.append(Document(
                text=text,
                metadata={"page_number": page_idx, "file_name": file_path.name},
            ))

        return documents

    def compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file content for incremental update detection."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
