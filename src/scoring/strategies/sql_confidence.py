from src.models.scoring import ConfidenceContext


class SQLConfidence:
    """Confidence based on SQL query success and result count."""

    def compute(self, ctx: ConfidenceContext) -> float:
        if not ctx.sql_success:
            return 0.0
        return 1.0 if ctx.sql_row_count > 0 else 0.5
