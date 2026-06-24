"""SemanticChunker — heading-aware document chunking for embedding."""

import hashlib
import re

import structlog
from llama_index.core.schema import Document
from transformers import AutoTokenizer

from knowledge_hub.config import Settings
from knowledge_hub.schemas import ChunkMetadata, DocumentChunk

logger = structlog.get_logger()


class SemanticChunker:
    """Splits documents into semantic chunks for embedding.

    Strategy:
    1. Split by markdown headings first (preserves heading chain)
    2. Within each section, split by paragraph boundaries
    3. Merge small paragraphs until approaching max_tokens
    4. Hard split at max_tokens for oversized elements (tables, code blocks)
    5. Overlap adjacent chunks by keeping the last paragraph when overlap > 0
    """

    def __init__(self, settings: Settings):
        self._max_tokens = settings.CHUNK_MAX_TOKENS
        self._overlap = settings.CHUNK_OVERLAP

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.EMBED_MODEL)
            self._count_tokens = lambda text: len(
                self._tokenizer.encode(text, add_special_tokens=False)
            )
        except Exception as e:
            logger.warning("tokenizer_load_failed_falling_back", error=str(e))
            self._tokenizer = None
            self._count_tokens = lambda text: max(1, len(text) // 4)

    def chunk(
        self, documents: list[Document], source_file: str, source_hash: str
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for doc in documents:
            if not doc.text or not doc.text.strip():
                continue
            sections = self._split_by_headings(doc.text)
            for heading_chain, section_text in sections:
                section_chunks = self._split_by_tokens(
                    section_text, heading_chain, source_file, source_hash
                )
                chunks.extend(section_chunks)
        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[list[str], str]]:
        """Split text by markdown headings, tracking the heading chain."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections: list[tuple[list[str], str]] = []
        current_headings: list[str] = []
        last_pos = 0

        for match in heading_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Capture text before this heading
            if last_pos < match.start():
                section_text = text[last_pos : match.start()].strip()
                if section_text:
                    sections.append((list(current_headings), section_text))

            # Update heading chain: truncate to parent level, then append
            current_headings = current_headings[: level - 1]
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
        self,
        text: str,
        heading_chain: list[str],
        source_file: str,
        source_hash: str,
    ) -> list[DocumentChunk]:
        """Split text into chunks respecting max_tokens, with overlap."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[DocumentChunk] = []
        current_texts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self._estimate_tokens(para)

            # Hard split for oversized single elements
            if para_tokens > self._max_tokens:
                # Flush current buffer first
                if current_texts:
                    chunks.append(
                        self._make_chunk(
                            "\n\n".join(current_texts),
                            heading_chain,
                            source_file,
                            source_hash,
                        )
                    )
                    current_texts = []
                    current_tokens = 0
                # Split oversized paragraph into sub-chunks
                sub_chunks = self._hard_split(
                    para, heading_chain, source_file, source_hash
                )
                chunks.extend(sub_chunks)
                continue

            if current_tokens + para_tokens > self._max_tokens and current_texts:
                chunks.append(
                    self._make_chunk(
                        "\n\n".join(current_texts),
                        heading_chain,
                        source_file,
                        source_hash,
                    )
                )
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
            chunks.append(
                self._make_chunk(
                    "\n\n".join(current_texts),
                    heading_chain,
                    source_file,
                    source_hash,
                )
            )

        return chunks

    def _hard_split(
        self,
        text: str,
        heading_chain: list[str],
        source_file: str,
        source_hash: str,
    ) -> list[DocumentChunk]:
        """Split an oversized text element into max_tokens-sized chunks."""
        # Use character-based estimation: ~4 chars per token
        chars_per_chunk = self._max_tokens * 4
        chunks = []
        for i in range(0, len(text), chars_per_chunk):
            sub_text = text[i:i + chars_per_chunk]
            chunks.append(self._make_chunk(sub_text, heading_chain, source_file, source_hash))
        return chunks

    def _make_chunk(
        self,
        text: str,
        heading_chain: list[str],
        source_file: str,
        source_hash: str,
    ) -> DocumentChunk:
        """Create a DocumentChunk with a deterministic ID."""
        raw_id = f"{source_file}|{'|'.join(heading_chain)}{text[:200]}"
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return max(1, len(text) // 4)
