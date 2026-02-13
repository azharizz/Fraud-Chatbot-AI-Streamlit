import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from src.core.llm_client import LLMClient
from src.models.source_type import SourceType
from src.models.scoring import ConfidenceContext
from src.scoring.strategies import compute_confidence


@pytest.fixture(scope="module")
def llm_client():
    return LLMClient(OpenAI())


# ---------------------------------------------------------------------------
# Confidence strategy tests (unit-level, no API calls)
# ---------------------------------------------------------------------------

class TestConfidenceStrategies:

    def test_sql_success_with_rows(self):
        ctx = ConfidenceContext(source_type=SourceType.SQL, sql_success=True, sql_row_count=10)
        assert compute_confidence(ctx) == 1.0

    def test_sql_success_no_rows(self):
        ctx = ConfidenceContext(source_type=SourceType.SQL, sql_success=True, sql_row_count=0)
        assert compute_confidence(ctx) == 0.5

    def test_sql_failure(self):
        ctx = ConfidenceContext(source_type=SourceType.SQL, sql_success=False)
        assert compute_confidence(ctx) == 0.0

    def test_rag_with_scores(self):
        ctx = ConfidenceContext(
            source_type=SourceType.RAG,
            similarity_scores=[0.8, 0.7, 0.9],
        )
        score = compute_confidence(ctx)
        assert 0.79 < score < 0.81

    def test_rag_no_scores(self):
        ctx = ConfidenceContext(source_type=SourceType.RAG)
        assert compute_confidence(ctx) == 0.5

    def test_combined(self):
        ctx = ConfidenceContext(
            source_type=SourceType.BOTH,
            sql_success=True,
            sql_row_count=5,
            similarity_scores=[0.8, 0.8],
        )
        score = compute_confidence(ctx)
        assert 0.8 < score < 1.0

    def test_error_source_type(self):
        ctx = ConfidenceContext(source_type=SourceType.ERROR)
        assert compute_confidence(ctx) == 0.5


# ---------------------------------------------------------------------------
# Quality scorer tests (require OpenAI API)
# ---------------------------------------------------------------------------

class TestQualityScorer:

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="No .env file found",
    )
    def test_quality_score_sql(self, llm_client):
        from src.scoring.quality import QualityScorer

        scorer = QualityScorer(llm_client)
        score = scorer.score(
            question="What is the fraud rate?",
            answer="The overall fraud rate is 0.58%, with 10,748 fraudulent transactions out of 1,852,394 total.",
            context="fraud_count: 10748, total: 1852394, rate: 0.58%",
            source_type=SourceType.SQL,
            sql_success=True,
            sql_row_count=1,
        )
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.faithfulness <= 1.0
        assert 0.0 <= score.relevance <= 1.0
        assert score.confidence == 1.0

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="No .env file found",
    )
    def test_quality_score_rag(self, llm_client):
        from src.scoring.quality import QualityScorer

        scorer = QualityScorer(llm_client)
        score = scorer.score(
            question="What is SCA?",
            answer="Strong Customer Authentication (SCA) is a multi-factor authentication requirement under PSD2.",
            context="SCA requires two of: knowledge, possession, inherence factors for electronic payments.",
            source_type=SourceType.RAG,
            similarity_scores=[0.85, 0.78],
        )
        assert 0.0 <= score.overall <= 1.0
        assert score.confidence > 0.7


# ---------------------------------------------------------------------------
# Validation tests (unit-level, no API calls)
# ---------------------------------------------------------------------------

class TestAnswerValidator:

    def test_sql_validation_grounded(self):
        from src.scoring.validation import AnswerValidator

        validator = AnswerValidator()
        passed, reason = validator.validate(
            answer="There were 10748 fraudulent transactions with a rate of 0.58%.",
            source_type=SourceType.SQL,
            sql_results=[{"fraud_count": 10748, "rate": 0.58}],
        )
        assert passed

    def test_rag_validation_grounded(self):
        from src.scoring.validation import AnswerValidator

        validator = AnswerValidator()
        passed, reason = validator.validate(
            answer="According to the report, SCA has reduced fraud rates by 30%.",
            source_type=SourceType.RAG,
            retrieved_chunks=["SCA has reduced fraud rates by 30% according to the EBA report."],
        )
        assert passed
