from src.models.scoring import ConfidenceContext
from src.scoring.strategies.sql_confidence import SQLConfidence
from src.scoring.strategies.rag_confidence import RAGConfidence


class CombinedConfidence:
    """Average of SQL and RAG confidence scores."""

    def compute(self, ctx: ConfidenceContext) -> float:
        rag_score = RAGConfidence().compute(ctx)
        sql_score = SQLConfidence().compute(ctx)
        return (rag_score + sql_score) / 2.0
