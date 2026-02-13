from src.models.scoring import ConfidenceContext
from src.models.source_type import SourceType
from src.scoring.strategies.base import ConfidenceStrategy
from src.scoring.strategies.sql_confidence import SQLConfidence
from src.scoring.strategies.rag_confidence import RAGConfidence
from src.scoring.strategies.combined_confidence import CombinedConfidence

STRATEGIES: dict[SourceType, ConfidenceStrategy] = {
    SourceType.SQL: SQLConfidence(),
    SourceType.RAG: RAGConfidence(),
    SourceType.BOTH: CombinedConfidence(),
}


def compute_confidence(ctx: ConfidenceContext) -> float:
    """Dispatch to the appropriate confidence strategy."""
    strategy = STRATEGIES.get(ctx.source_type)
    if strategy is None:
        return 0.5
    return strategy.compute(ctx)
