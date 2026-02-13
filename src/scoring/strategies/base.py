from typing import Protocol

from src.models.scoring import ConfidenceContext


class ConfidenceStrategy(Protocol):
    """Interface for confidence scoring strategies."""

    def compute(self, ctx: ConfidenceContext) -> float: ...
