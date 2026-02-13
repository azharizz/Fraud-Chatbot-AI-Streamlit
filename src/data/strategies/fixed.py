from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.data.strategies.base import Pages, ChunkList
from src.models.chunks import ChunkMetadata


class FixedChunking:
    """Split pages into fixed-size overlapping chunks."""

    def chunk(self, pages: Pages, source: str) -> ChunkList:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: ChunkList = []
        chunk_id = 0
        for page_num, text in pages:
            for split_text in splitter.split_text(text):
                chunks.append({
                    "text": split_text,
                    "metadata": ChunkMetadata(
                        source=source, page=page_num, chunk_id=chunk_id,
                    ),
                })
                chunk_id += 1
        return chunks
