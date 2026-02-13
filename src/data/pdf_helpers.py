import logging
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
from openai import OpenAI

from src.core.config import EMBEDDING_MODEL
from src.models.chunks import ChunkMetadata

logger = logging.getLogger(__name__)


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = [
        (page_num + 1, doc[page_num].get_text())
        for page_num in range(len(doc))
        if doc[page_num].get_text().strip()
    ]
    doc.close()
    return pages


def embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """Batch-embed texts via OpenAI and return normalized numpy array."""
    embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend([item.embedding for item in response.data])
        logger.info(
            "Embedded batch %d/%d",
            i // batch_size + 1,
            (len(texts) - 1) // batch_size + 1,
        )
    arr = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def coerce_metadata(metadata: Any) -> ChunkMetadata:
    """Convert legacy metadata formats to Pydantic ChunkMetadata."""
    if isinstance(metadata, ChunkMetadata):
        return metadata
    if isinstance(metadata, dict):
        return ChunkMetadata(**metadata)
    return ChunkMetadata(
        source=str(getattr(metadata, "source", "")),
        page=int(getattr(metadata, "page", 0)),
        chunk_id=int(getattr(metadata, "chunk_id", 0)),
        section=str(getattr(metadata, "section", "")),
    )
