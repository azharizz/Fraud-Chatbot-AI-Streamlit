import logging
from typing import Any

import duckdb

from src.agent.prompts import SQL_SYSTEM_PROMPT, SQL_ERROR_CORRECTION_PROMPT, format_sql_few_shot
from src.core.config import MAX_SQL_RETRIES, PII_COLUMNS
from src.core.llm_client import LLMClient
from src.data.database import FraudDatabase
from src.models.tools import SQLToolResult

logger = logging.getLogger(__name__)


class SQLTool:
    """Text-to-SQL pipeline: generate SQL from questions, execute, mask PII."""

    def __init__(self, llm_client: LLMClient, database: FraudDatabase) -> None:
        self._llm = llm_client
        self._db = database

    def run(self, question: str) -> SQLToolResult:
        """Execute the Text-to-SQL pipeline. Returns typed SQLToolResult."""
        system_prompt = self._build_prompt()
        sql = self._generate_sql(system_prompt, question)
        logger.info("Generated SQL:\n%s", sql)

        result = self._db.execute_query(sql)

        if not result.success and MAX_SQL_RETRIES > 0:
            logger.info("SQL failed, attempting self-correction...")
            error_prompt = SQL_ERROR_CORRECTION_PROMPT.format(
                error=result.error, failed_sql=sql,
            )
            sql = self._generate_sql(system_prompt, question, error_context=error_prompt)
            logger.info("Corrected SQL:\n%s", sql)
            result = self._db.execute_query(sql)

        if result.success:
            masked = self._mask_pii(result.columns, result.rows)
            return SQLToolResult(
                success=True,
                sql_query=sql,
                columns=result.columns,
                rows=masked,
                row_count=result.row_count,
            )

        return SQLToolResult(success=False, sql_query=sql, error=result.error)

    def _build_prompt(self) -> str:
        """Build the SQL system prompt with schema, sample rows, stats, and few-shot."""
        schema = self._db.get_schema()
        sample = self._db.get_sample_rows(n=3)
        few_shot = format_sql_few_shot()
        stats = self._get_column_stats()
        return SQL_SYSTEM_PROMPT.format(schema=schema, sample_rows=sample) + "\n" + stats + "\n" + few_shot

    def _get_column_stats(self) -> str:
        """Fetch column statistics from the database for prompt context."""
        try:
            con = self._db.connection
            lines = ["\n**Column statistics**:"]

            date_range = con.execute(
                "SELECT MIN(trans_date_trans_time)::DATE, MAX(trans_date_trans_time)::DATE FROM transactions"
            ).fetchone()
            lines.append(f"- Date range: {date_range[0]} to {date_range[1]}")

            total = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            frauds = con.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1").fetchone()[0]
            lines.append(f"- Total transactions: {total:,}")
            lines.append(f"- Fraudulent: {frauds:,} ({100.0 * frauds / total:.2f}%)")

            cats = con.execute("SELECT DISTINCT category FROM transactions ORDER BY category").fetchall()
            lines.append(f"- Categories ({len(cats)}): {', '.join(c[0] for c in cats)}")

            amounts = con.execute(
                "SELECT ROUND(MIN(amt),2), ROUND(MAX(amt),2), ROUND(AVG(amt),2) FROM transactions"
            ).fetchone()
            lines.append(f"- Amount range: ${amounts[0]} – ${amounts[1]} (avg: ${amounts[2]})")

            months = con.execute(
                "SELECT MIN(transaction_month), MAX(transaction_month) FROM transactions"
            ).fetchone()
            lines.append(f"- transaction_month range: '{months[0]}' to '{months[1]}' (VARCHAR, YYYY-MM format)")

            return "\n".join(lines)
        except Exception as exc:
            logger.warning("Could not get column stats: %s", exc)
            return ""

    def _generate_sql(
        self,
        system_prompt: str,
        question: str,
        error_context: str | None = None,
    ) -> str:
        """Call LLM to generate a SQL query."""
        user_content = error_context or question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        sql = self._llm.chat(messages)

        if sql.startswith("```"):
            lines = [line for line in sql.split("\n") if not line.startswith("```")]
            sql = "\n".join(lines).strip()
        return sql

    @staticmethod
    def _mask_pii(columns: list[str], rows: list[tuple]) -> list[dict[str, Any]]:
        """Mask PII columns in query results."""
        masked: list[dict[str, Any]] = []
        for row in rows:
            record = {
                col: ("***MASKED***" if col.lower() in PII_COLUMNS else val)
                for col, val in zip(columns, row)
            }
            masked.append(record)
        return masked
