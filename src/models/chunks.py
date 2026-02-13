from pydantic import BaseModel, ConfigDict


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk extracted from a PDF."""

    model_config = ConfigDict(frozen=True)

    source: str
    page: int
    chunk_id: int
    section: str = ""


class SearchResult(BaseModel):
    """A single search result with text, metadata, and similarity score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    metadata: ChunkMetadata
    score: float
