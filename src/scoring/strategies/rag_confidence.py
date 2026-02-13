from src.models.scoring import ConfidenceContext


class RAGConfidence:
    """Confidence based on average similarity scores from retrieved chunks."""

    def compute(self, ctx: ConfidenceContext) -> float:
        if not ctx.similarity_scores:
            return 0.5
        avg = sum(ctx.similarity_scores) / len(ctx.similarity_scores)
        return max(0.0, min(1.0, avg))
