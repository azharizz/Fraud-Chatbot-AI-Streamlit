from typing import Any

from pydantic import BaseModel


class QueryResult(BaseModel):
    """Result from a raw SQL execution in the database layer."""

    success: bool
    columns: list[str] = []
    rows: list[Any] = []
    row_count: int = 0
    error: str | None = None


class SQLToolResult(BaseModel):
    """Result from the Text-to-SQL pipeline."""

    success: bool
    sql_query: str = ""
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    error: str | None = None


class RAGToolResult(BaseModel):
    """Result from the RAG retrieval pipeline."""

    success: bool
    answer: str = ""
    retrieved_chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    similarity_scores: list[float] = []
    error: str | None = None
