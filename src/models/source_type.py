from enum import Enum


class SourceType(str, Enum):
    SQL = "sql"
    RAG = "rag"
    BOTH = "both"
    ERROR = "error"
