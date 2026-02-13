import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

from src.core.llm_client import LLMClient
from src.data.database import FraudDatabase
from src.agent.sql_tool import SQLTool


@pytest.fixture(scope="module")
def db():
    return FraudDatabase.connect()


@pytest.fixture(scope="module")
def llm_client():
    return LLMClient(OpenAI())


@pytest.fixture(scope="module")
def sql_tool(llm_client, db):
    return SQLTool(llm_client, db)


class TestDatabase:

    def test_connection(self, db):
        result = db.connection.execute("SELECT COUNT(*) FROM transactions").fetchone()
        assert result[0] > 0

    def test_schema_available(self, db):
        schema = db.get_schema()
        assert "transactions" in schema
        assert "is_fraud" in schema

    def test_validate_select(self):
        assert FraudDatabase.validate_query("SELECT * FROM transactions") is None

    def test_validate_reject_delete(self):
        result = FraudDatabase.validate_query("DELETE FROM transactions")
        assert result is not None

    def test_execute_query_basic(self, db):
        result = db.execute_query("SELECT COUNT(*) AS cnt FROM transactions")
        assert result.success
        assert result.row_count == 1
        assert result.columns == ["cnt"]

    def test_execute_query_fraud_rate(self, db):
        result = db.execute_query(
            "SELECT COUNT(*) FILTER (WHERE is_fraud = 1) AS fraud_count, "
            "COUNT(*) AS total FROM transactions"
        )
        assert result.success
        assert result.row_count == 1

    def test_execute_invalid_query(self, db):
        result = db.execute_query("SELECT * FROM nonexistent_table")
        assert not result.success
        assert result.error


class TestSQLTool:

    def test_sql_tool_basic(self, sql_tool):
        result = sql_tool.run("What is the total number of transactions?")
        assert result.success
        assert result.sql_query
        assert result.row_count >= 1

    def test_sql_tool_fraud_rate(self, sql_tool):
        result = sql_tool.run("What is the overall fraud rate?")
        assert result.success
        assert result.rows

    def test_sql_tool_monthly_trend(self, sql_tool):
        result = sql_tool.run("Show the monthly fraud count over time")
        assert result.success
        assert result.row_count > 1

    def test_pii_masking(self):
        columns = ["name", "cc_num", "amount"]
        rows = [("John", 1234567890, 100.50)]
        masked = SQLTool._mask_pii(columns, rows)
        assert masked[0]["cc_num"] == "***MASKED***"
        assert masked[0]["amount"] == 100.50
