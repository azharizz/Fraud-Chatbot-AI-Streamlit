import logging

from src.core.llm_client import LLMClient
from src.agent.prompts import SYNTHESIS_PROMPT
from src.models.tools import SQLToolResult, RAGToolResult

logger = logging.getLogger(__name__)


class ResultSynthesizer:
    """Synthesize SQL and RAG results into a single unified answer."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def synthesize(
        self,
        question: str,
        sql: SQLToolResult,
        rag: RAGToolResult,
    ) -> str:
        """Returns the synthesized answer, or empty string on failure."""
        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            sql_context=self._format_sql_context(sql),
            rag_context=self._format_rag_context(rag),
        )
        try:
            return self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return ""

    @staticmethod
    def _format_sql_context(sql: SQLToolResult) -> str:
        """Format SQL results as context for the synthesis prompt."""
        if not sql.success:
            return "No SQL data available."
        lines = [f"Query: {sql.sql_query}", f"Results ({sql.row_count} rows):"]
        for row in sql.rows[:20]:
            lines.append(" | ".join(str(row.get(c, "")) for c in sql.columns))
        return "\n".join(lines)

    @staticmethod
    def _format_rag_context(rag: RAGToolResult) -> str:
        """Format RAG results as context for the synthesis prompt."""
        if not rag.success:
            return "No document results available."
        parts = [rag.answer]
        if rag.retrieved_chunks:
            parts.append("\nRelevant excerpts:")
            for chunk in rag.retrieved_chunks[:3]:
                parts.append(f"- {chunk[:200]}...")
        return "\n".join(parts)
