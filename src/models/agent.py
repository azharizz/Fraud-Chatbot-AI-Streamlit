from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from src.models.source_type import SourceType

if TYPE_CHECKING:
    import duckdb
    import faiss
    from openai import OpenAI


class AgentDeps(BaseModel):
    """Dependencies injected into the PydanticAI agent at runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    con: duckdb.DuckDBPyConnection
    openai_client: OpenAI
    faiss_index: faiss.IndexFlatIP
    chunks: list[dict[str, Any]]
    tool_outputs: dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Structured response returned by the router agent."""

    answer: str
    source_type: SourceType = SourceType.ERROR
    sql_query: str | None = None
    sql_results: list[dict[str, Any]] | None = None
    sql_columns: list[str] | None = None
    retrieved_chunks: list[str] | None = None
    similarity_scores: list[float] | None = None
    sources: list[dict[str, Any]] | None = None
    error: str | None = None
