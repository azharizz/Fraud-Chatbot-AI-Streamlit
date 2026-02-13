import logging
import re
from typing import Protocol

from src.models.source_type import SourceType

logger = logging.getLogger(__name__)

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "that", "for"}


class ValidationStrategy(Protocol):
    """Interface for source-specific answer validation."""

    def validate(self, answer: str, **kwargs) -> str | None: ...


class SQLValidator:
    """Check that numbers cited in the answer exist in the SQL results."""

    def validate(self, answer: str, **kwargs) -> str | None:
        sql_results: list[dict] = kwargs.get("sql_results", [])
        if not sql_results:
            return None

        answer_numbers = set()
        for match in re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', answer):
            try:
                answer_numbers.add(float(match.replace(",", "")))
            except ValueError:
                pass

        if not answer_numbers:
            return None

        data_numbers = set()
        for row in sql_results:
            for val in row.values():
                if isinstance(val, (int, float)):
                    data_numbers.update({float(val), round(float(val), 2), round(float(val), 4)})
                elif isinstance(val, str):
                    try:
                        data_numbers.add(float(val))
                    except (ValueError, TypeError):
                        pass

        ungrounded = []
        for num in answer_numbers:
            if num < 1 or num <= 10:
                continue
            matched = any(
                abs(num - d) < 0.1 or abs(num - d) / max(abs(d), 1) < 0.01
                for d in data_numbers
            ) if data_numbers else False
            if not matched:
                ungrounded.append(num)

        if ungrounded and len(ungrounded) > len(answer_numbers) * 0.5:
            samples = ", ".join(str(int(n)) for n in list(ungrounded)[:3])
            return f"Some numbers in the answer may not match query results: {samples}"

        return None


class RAGValidator:
    """Check that factual claims are supported by retrieved chunks."""

    def validate(self, answer: str, **kwargs) -> str | None:
        retrieved_chunks: list[str] = kwargs.get("retrieved_chunks", [])
        if not retrieved_chunks:
            return None

        chunks_joined = " ".join(c.lower() for c in retrieved_chunks)

        claims = re.findall(
            r'(?:according to|reported|found that|stated that)\s+(.{20,80}?)(?:\.|,|\n)',
            answer.lower(),
        )

        ungrounded = []
        for claim in claims:
            content_words = set(claim.split()) - _STOPWORDS
            if not content_words:
                continue
            coverage = sum(1 for w in content_words if w in chunks_joined) / len(content_words)
            if coverage < 0.3:
                ungrounded.append(claim.strip()[:50])

        if ungrounded:
            return f'Some claims may not be supported by source documents: "{ungrounded[0]}..."'

        return None


_VALIDATORS: dict[SourceType, list[ValidationStrategy]] = {
    SourceType.SQL: [SQLValidator()],
    SourceType.RAG: [RAGValidator()],
    SourceType.BOTH: [SQLValidator(), RAGValidator()],
}


class AnswerValidator:
    """Validate that an answer is grounded in its source data."""

    def validate(
        self,
        answer: str,
        source_type: SourceType | str,
        sql_results: list[dict] | None = None,
        retrieved_chunks: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Returns (passed, reason)."""
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        validators = _VALIDATORS.get(source_type, [])
        issues: list[str] = []

        for validator in validators:
            issue = validator.validate(
                answer,
                sql_results=sql_results or [],
                retrieved_chunks=retrieved_chunks or [],
            )
            if issue:
                issues.append(issue)

        if issues:
            return False, "; ".join(issues)
        return True, "Answer is grounded in source data"
