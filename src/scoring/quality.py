import json
import logging

import numpy as np

from src.agent.prompts import FAITHFULNESS_PROMPT
from src.core.llm_client import LLMClient
from src.models.scoring import ConfidenceContext, QualityScore
from src.models.source_type import SourceType
from src.scoring.strategies import compute_confidence

logger = logging.getLogger(__name__)


class QualityScorer:
    """Compute overall quality scores for chatbot responses.

    Weights: 50% faithfulness, 30% relevance, 20% confidence.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        source_type: SourceType | str,
        similarity_scores: list[float] | None = None,
        sql_success: bool = False,
        sql_row_count: int = 0,
    ) -> QualityScore:
        """Compute overall quality score for a chatbot response."""
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        faithfulness, faith_reason = self._score_faithfulness(question, answer, context)
        relevance = self._score_relevance(question, answer)

        confidence = compute_confidence(ConfidenceContext(
            source_type=source_type,
            similarity_scores=similarity_scores,
            sql_success=sql_success,
            sql_row_count=sql_row_count,
        ))

        overall = 0.5 * faithfulness + 0.3 * relevance + 0.2 * confidence

        return QualityScore(
            faithfulness=round(faithfulness, 4),
            faithfulness_reason=faith_reason,
            relevance=round(relevance, 4),
            confidence=round(confidence, 4),
            overall=round(overall, 4),
        )

    def _score_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> tuple[float, str]:
        """LLM-as-judge faithfulness scoring. Returns (score, reason)."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context, question=question, answer=answer,
        )
        try:
            raw = self._llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            if raw.startswith("```"):
                lines = [line for line in raw.split("\n") if not line.startswith("```")]
                raw = "\n".join(lines).strip()

            result = json.loads(raw)
            score = max(0.0, min(1.0, float(result.get("score", 0.5))))
            reason = result.get("reason", "No reason provided")
            return score, reason

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse faithfulness score: %s", exc)
            return 0.5, "Could not evaluate faithfulness"

    def _score_relevance(self, question: str, answer: str) -> float:
        """Cosine similarity between question and answer embeddings."""
        try:
            vecs = self._llm.embed([question, answer])
            q_vec = np.array(vecs[0], dtype=np.float32)
            a_vec = np.array(vecs[1], dtype=np.float32)

            norm = np.linalg.norm(q_vec) * np.linalg.norm(a_vec)
            if norm == 0:
                return 0.0
            similarity = float(np.dot(q_vec, a_vec) / norm)
            return max(0.0, min(1.0, similarity))

        except Exception as exc:
            logger.warning("Relevance scoring failed: %s", exc)
            return 0.5
