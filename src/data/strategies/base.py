from typing import Any, Protocol

Pages = list[tuple[int, str]]
ChunkList = list[dict[str, Any]]


class ChunkingStrategy(Protocol):
    """Interface for text chunking strategies."""

    def chunk(self, pages: Pages, source: str) -> ChunkList: ...
