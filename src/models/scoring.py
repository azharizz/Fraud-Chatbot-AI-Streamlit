from pydantic import BaseModel

from src.models.source_type import SourceType


class QualityScore(BaseModel):
    """Quality score breakdown for a chatbot response."""

    faithfulness: float
    faithfulness_reason: str
    relevance: float
    confidence: float
    overall: float
    validation_passed: bool | None = None
    validation_reason: str = ""


class ConfidenceContext(BaseModel):
    """Input context for confidence scoring strategies."""

    source_type: SourceType
    similarity_scores: list[float] | None = None
    sql_success: bool = False
    sql_row_count: int = 0
