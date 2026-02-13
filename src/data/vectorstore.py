from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

from src.core.config import EMBEDDING_MODEL, CHUNKING_MODE
from src.data.strategies import chunk_pages
from src.data.pdf_helpers import extract_pdf_pages, embed_texts, coerce_metadata
from src.models.chunks import ChunkMetadata, SearchResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index.bin"
CHUNKS_PATH = PROCESSED_DIR / "chunks.pkl"

EMBEDDING_DIM = 1536

PDF_SOURCES = {
    "Bhatla.pdf": "bhatla",
    "EBA_ECB_2024_Report.pdf": "eba_ecb_2024",
}


class VectorStore:
    """FAISS-backed vector store for PDF chunk retrieval."""

    def __init__(self, index: Any, chunks: list[dict[str, Any]]) -> None:
        self._index = index
        self._chunks = chunks

    @property
    def index(self) -> Any:
        return self._index

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return self._chunks

    @classmethod
    def from_pdfs(cls) -> VectorStore:
        """Parse PDFs, chunk, embed, and store in FAISS. Returns a new VectorStore."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        client = OpenAI()
        all_chunks: list[dict[str, Any]] = []
        mode = os.environ.get("CHUNKING_MODE", CHUNKING_MODE)

        for filename, source_key in PDF_SOURCES.items():
            pdf_path = RAW_DIR / filename
            if not pdf_path.exists():
                logger.warning("PDF not found: %s, skipping", pdf_path)
                continue

            logger.info("Parsing %s (mode=%s)...", filename, mode)
            pages = extract_pdf_pages(pdf_path)
            chunks = chunk_pages(pages, source_key, mode=mode)
            logger.info("  %d pages -> %d chunks", len(pages), len(chunks))
            all_chunks.extend(chunks)

        if not all_chunks:
            raise FileNotFoundError("No PDF files found in data/raw/")

        logger.info("Embedding %d total chunks...", len(all_chunks))
        texts = [c["text"] for c in all_chunks]
        vectors = embed_texts(texts, client)

        logger.info("Building FAISS index (dim=%d)...", EMBEDDING_DIM)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(vectors)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)

        logger.info("Saved FAISS index and %d chunks", len(all_chunks))
        return cls(index, all_chunks)

    @classmethod
    def load(cls) -> VectorStore:
        """Load a previously saved VectorStore from disk."""
        if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run 'python scripts/ingest.py' first."
            )
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded FAISS index (%d vectors) and %d chunks", index.ntotal, len(chunks))
        return cls(index, chunks)

    def search(
        self,
        query: str,
        client: OpenAI,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search the vector store for chunks matching the query."""
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        query_vec = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        search_k = top_k * 3 if source_filter else top_k
        scores, indices = self._index.search(query_vec, min(search_k, self._index.ntotal))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self._chunks[idx]
            metadata = chunk["metadata"]
            meta = coerce_metadata(metadata)

            if source_filter and meta.source != source_filter:
                continue

            results.append(SearchResult(text=chunk["text"], metadata=meta, score=float(score)))
            if len(results) >= top_k:
                break

        return results
