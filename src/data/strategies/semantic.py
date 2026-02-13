import logging
import re

from src.core.config import SEMANTIC_MIN_CHUNK, SEMANTIC_MAX_CHUNK
from src.data.strategies.base import Pages, ChunkList
from src.models.chunks import ChunkMetadata

logger = logging.getLogger(__name__)


class SemanticChunking:
    """Split pages by paragraph boundaries with section-aware metadata."""

    @staticmethod
    def _detect_section_header(text: str) -> str:
        """Detect if the first few lines contain a section header."""
        lines = text.strip().split("\n")
        for line in lines[:3]:
            line = line.strip()
            if line and len(line) < 120 and not line.endswith(".") and not line.endswith(","):
                if re.match(r'^(\d+\.?\s|[A-Z]{2,}|Chapter|Section|Part\s)', line) or len(line) < 60:
                    return line
        return ""

    def chunk(self, pages: Pages, source: str) -> ChunkList:
        chunks: ChunkList = []
        chunk_id = 0
        current_section = ""

        for page_num, text in pages:
            paragraphs = re.split(r'\n\s*\n', text)
            buffer = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                header = self._detect_section_header(para)
                if header and len(para) < 120:
                    current_section = header

                if buffer and len(buffer) + len(para) > SEMANTIC_MAX_CHUNK:
                    chunks.append({
                        "text": buffer.strip(),
                        "metadata": ChunkMetadata(
                            source=source, page=page_num,
                            chunk_id=chunk_id, section=current_section,
                        ),
                    })
                    chunk_id += 1
                    buffer = ""

                buffer += para + "\n\n"

                if len(buffer) > SEMANTIC_MAX_CHUNK:
                    sentences = re.split(r'(?<=[.!?])\s+', buffer)
                    sentence_buffer = ""
                    for sent in sentences:
                        if sentence_buffer and len(sentence_buffer) + len(sent) > SEMANTIC_MAX_CHUNK:
                            chunks.append({
                                "text": sentence_buffer.strip(),
                                "metadata": ChunkMetadata(
                                    source=source, page=page_num,
                                    chunk_id=chunk_id, section=current_section,
                                ),
                            })
                            chunk_id += 1
                            sentence_buffer = ""
                        sentence_buffer += sent + " "
                    buffer = sentence_buffer

            if buffer.strip():
                if len(buffer.strip()) < SEMANTIC_MIN_CHUNK and chunks and chunks[-1]["metadata"].source == source:
                    chunks[-1]["text"] += "\n\n" + buffer.strip()
                else:
                    chunks.append({
                        "text": buffer.strip(),
                        "metadata": ChunkMetadata(
                            source=source, page=page_num,
                            chunk_id=chunk_id, section=current_section,
                        ),
                    })
                    chunk_id += 1

        return chunks
