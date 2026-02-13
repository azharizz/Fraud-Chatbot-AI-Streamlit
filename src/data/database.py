import logging
import re
from pathlib import Path

import duckdb

from src.models.tools import QueryResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "processed" / "fraud.duckdb"

MAX_QUERY_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 10
_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_CSV_COLUMNS = {
    "Unnamed: 0": "INTEGER",
    "trans_date_trans_time": "VARCHAR",
    "cc_num": "BIGINT",
    "merchant": "VARCHAR",
    "category": "VARCHAR",
    "amt": "DOUBLE",
    "first": "VARCHAR",
    "last": "VARCHAR",
    "gender": "VARCHAR",
    "street": "VARCHAR",
    "city": "VARCHAR",
    "state": "VARCHAR",
    "zip": "INTEGER",
    "lat": "DOUBLE",
    "long": "DOUBLE",
    "city_pop": "INTEGER",
    "job": "VARCHAR",
    "dob": "VARCHAR",
    "trans_num": "VARCHAR",
    "unix_time": "BIGINT",
    "merch_lat": "DOUBLE",
    "merch_long": "DOUBLE",
    "is_fraud": "INTEGER",
}


class FraudDatabase:
    """DuckDB-backed database for fraud transaction data."""

    _SCHEMA_DESCRIPTION = """Table: transactions
Columns:
- trans_date_trans_time (TIMESTAMP): Date and time of the transaction
- cc_num (BIGINT): Credit card number
- merchant (VARCHAR): Merchant name (prefixed with "fraud_")
- category (VARCHAR): Transaction category (e.g., grocery_pos, shopping_net, misc_net, etc.)
- amt (DOUBLE): Transaction amount in USD
- first (VARCHAR): Cardholder first name
- last (VARCHAR): Cardholder last name
- gender (VARCHAR): Cardholder gender (M or F)
- street (VARCHAR): Cardholder street address
- city (VARCHAR): Cardholder city
- state (VARCHAR): Cardholder US state code
- zip (INTEGER): Cardholder ZIP code
- lat (DOUBLE): Cardholder latitude
- long (DOUBLE): Cardholder longitude
- city_pop (INTEGER): Population of cardholder's city
- job (VARCHAR): Cardholder occupation
- dob (VARCHAR): Cardholder date of birth
- trans_num (VARCHAR): Unique transaction identifier
- unix_time (BIGINT): Unix timestamp of the transaction
- merch_lat (DOUBLE): Merchant latitude
- merch_long (DOUBLE): Merchant longitude
- is_fraud (INTEGER): Fraud label (0 = legitimate, 1 = fraudulent)
- transaction_month (VARCHAR): Pre-computed 'YYYY-MM' month string
- transaction_hour (INTEGER): Pre-computed hour of day (0-23)

Date range: 2019-01-01 to 2020-12-31
Total rows: ~1,852,394
Fraud rate: ~0.6%
SQL dialect: DuckDB (use strftime for date formatting, FILTER clause for conditional aggregation)"""

    _SELECT_COLUMNS = (
        "CAST(trans_date_trans_time AS TIMESTAMP) AS trans_date_trans_time, "
        "cc_num, merchant, category, amt, first, last, gender, "
        "street, city, state, zip, lat, long, city_pop, job, dob, "
        "trans_num, unix_time, merch_lat, merch_long, is_fraud, "
        "strftime(CAST(trans_date_trans_time AS TIMESTAMP), '%Y-%m') AS transaction_month, "
        "EXTRACT(HOUR FROM CAST(trans_date_trans_time AS TIMESTAMP)) AS transaction_hour"
    )

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    @classmethod
    def connect(cls, read_only: bool = True) -> "FraudDatabase":
        """Create a new FraudDatabase with a connection to the default DB path."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(DB_PATH), read_only=read_only)
        return cls(con)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    def ingest_csv(self) -> int:
        """Load CSV files into the transactions table. Returns row count."""
        logger.info("Dropping existing transactions table if present")
        self._con.execute("DROP TABLE IF EXISTS transactions")

        csv_files = [RAW_DIR / "fraudTrain.csv", RAW_DIR / "fraudTest.csv"]
        for f in csv_files:
            if not f.exists():
                raise FileNotFoundError(f"CSV file not found: {f}")

        columns_spec = ", ".join(f"'{k}': '{v}'" for k, v in _CSV_COLUMNS.items())

        for i, csv_file in enumerate(csv_files):
            if i == 0:
                sql = (
                    f"CREATE TABLE transactions AS SELECT {self._SELECT_COLUMNS} "
                    f"FROM read_csv_auto(?, header=true, columns={{{columns_spec}}})"
                )
            else:
                sql = (
                    f"INSERT INTO transactions SELECT {self._SELECT_COLUMNS} "
                    f"FROM read_csv_auto(?, header=true, columns={{{columns_spec}}})"
                )
            self._con.execute(sql, [str(csv_file)])
            logger.info("Loaded %s into transactions table", csv_file.name)

        row_count = self._con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        logger.info("Total rows ingested: %s", f"{row_count:,}")
        return row_count

    def get_schema(self) -> str:
        """Return a formatted table schema string for LLM prompts."""
        return self._SCHEMA_DESCRIPTION

    def get_sample_rows(self, n: int = 5) -> str:
        """Return formatted sample rows for LLM prompts."""
        result = self._con.execute(f"SELECT * FROM transactions LIMIT {n}").fetchdf()
        return result.to_string(index=False)

    @staticmethod
    def validate_query(sql: str) -> str | None:
        """Returns error message if query is invalid, None if OK."""
        stripped = sql.strip().rstrip(";").strip()
        if not stripped.upper().startswith("SELECT"):
            return "Only SELECT queries are allowed."
        if _BLOCKED_KEYWORDS.search(stripped):
            return "Query contains blocked keywords."
        return None

    def execute_query(self, sql: str) -> QueryResult:
        """Execute a validated SQL query. Returns typed QueryResult."""
        error = self.validate_query(sql)
        if error:
            return QueryResult(success=False, error=error)

        if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            sql = sql.rstrip().rstrip(";") + f" LIMIT {MAX_QUERY_ROWS}"

        try:
            result = self._con.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return QueryResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
            )
        except Exception as exc:
            logger.warning("SQL execution failed: %s", exc)
            return QueryResult(success=False, error=str(exc))
