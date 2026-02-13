import os

MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = "text-embedding-3-small"

OPENAI_TIMEOUT: int = 30
MAX_API_RETRIES: int = 2

MAX_SQL_RETRIES: int = 1
MAX_QUERY_ROWS: int = 1000
QUERY_TIMEOUT_SECONDS: int = 10
PII_COLUMNS: set[str] = {"cc_num", "first", "last", "street"}

DEDUP_SIMILARITY_THRESHOLD: float = 0.95

CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "200"))
SEMANTIC_MIN_CHUNK: int = int(os.environ.get("SEMANTIC_MIN_CHUNK", "100"))
SEMANTIC_MAX_CHUNK: int = int(os.environ.get("SEMANTIC_MAX_CHUNK", "1500"))
CHUNKING_MODE: str = os.environ.get("CHUNKING_MODE", "semantic")

MIN_QUESTION_LENGTH: int = 3
MAX_QUESTION_LENGTH: int = 2000
