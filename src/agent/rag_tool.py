import logging
from typing import Any

from src.agent.prompts import RAG_GENERATION_PROMPT
from src.core.config import DEDUP_SIMILARITY_THRESHOLD
from src.core.llm_client import LLMClient
from src.data.vectorstore import VectorStore
from src.models.tools import RAGToolResult
from src.models.chunks import SearchResult

logger = logging.getLogger(__name__)

SOURCE_DISPLAY_NAMES = {
    "bhatla": "Understanding Credit Card Frauds (Bhatla et al.)",
    "eba_ecb_2024": "2024 Report on Payment Fraud (EBA/ECB)",
}

SOURCE_KEYWORDS: dict[str, list[str]] = {
    "bhatla": ["bhatla", "bhatla paper", "understanding credit card"],
    "eba_ecb_2024": [
        "eba", "ecb", "eea", "sca", "psd2", "cross-border",
        "h1 2023", "h2 2022", "2024 report", "payment fraud report",
    ],
}


class RAGTool:
    """RAG pipeline: retrieve relevant chunks and generate cited answers."""

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore) -> None:
        self._llm = llm_client
        self._store = vector_store

    def run(
        self,
        question: str,
        client: Any,
        top_k: int = 5,
    ) -> RAGToolResult:
        """Execute the RAG pipeline. Returns typed RAGToolResult."""
        try:
            source_filter = self._detect_source_filter(question)
            if source_filter:
                logger.info("Detected source filter: %s", source_filter)

            results = self._store.search(
                query=question, client=client,
                top_k=top_k, source_filter=source_filter,
            )

            if not results:
                return RAGToolResult(
                    success=True,
                    answer="I couldn't find relevant information in the available documents to answer this question.",
                )

            results = self._deduplicate(results)

            avg_score = sum(r.score for r in results) / len(results)
            if avg_score < 0.3:
                logger.warning("Low average similarity score: %.3f", avg_score)

            context = self._format_context(results)
            answer = self._generate_answer(question, context)

            return RAGToolResult(
                success=True,
                answer=answer,
                retrieved_chunks=[r.text for r in results],
                sources=[
                    {
                        "source": SOURCE_DISPLAY_NAMES.get(r.metadata.source, r.metadata.source),
                        "page": r.metadata.page,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ],
                similarity_scores=[r.score for r in results],
            )

        except Exception as exc:
            logger.error("RAG tool error: %s", exc, exc_info=True)
            return RAGToolResult(success=False, error=str(exc))

    @staticmethod
    def _detect_source_filter(question: str) -> str | None:
        """Detect which PDF source to filter by based on question keywords."""
        q_lower = question.lower()
        for source, keywords in SOURCE_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return source
        return None

    @staticmethod
    def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        """Remove near-duplicate chunks based on word overlap."""
        if len(results) <= 1:
            return results

        unique: list[SearchResult] = [results[0]]
        for r in results[1:]:
            r_words = set(r.text.lower().split())
            is_dup = any(
                r_words and set(u.text.lower().split())
                and len(r_words & set(u.text.lower().split())) / min(len(r_words), len(set(u.text.lower().split())))
                > DEDUP_SIMILARITY_THRESHOLD
                for u in unique
            )
            if not is_dup:
                unique.append(r)
        return unique

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format search results as numbered context for the LLM prompt."""
        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            source_name = SOURCE_DISPLAY_NAMES.get(r.metadata.source, r.metadata.source)
            parts.append(
                f"[{i}] [Source: {source_name}, Page {r.metadata.page}] "
                f"(relevance: {r.score:.3f})\n{r.text}"
            )
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Call LLM to generate an answer from the retrieved context."""
        prompt = RAG_GENERATION_PROMPT.format(context=context, question=question)
        return self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
