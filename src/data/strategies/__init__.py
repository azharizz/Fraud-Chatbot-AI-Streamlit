import logging

from src.core.config import CHUNKING_MODE
from src.data.strategies.base import ChunkingStrategy, Pages, ChunkList
from src.data.strategies.fixed import FixedChunking
from src.data.strategies.semantic import SemanticChunking

logger = logging.getLogger(__name__)

STRATEGIES: dict[str, ChunkingStrategy] = {
    "fixed": FixedChunking(),
    "semantic": SemanticChunking(),
}


def chunk_pages(pages: Pages, source: str, mode: str | None = None) -> ChunkList:
    """Dispatch to the appropriate chunking strategy."""
    _mode = mode or CHUNKING_MODE
    strategy = STRATEGIES.get(_mode)
    if strategy is None:
        logger.warning("Unknown chunking mode '%s', falling back to fixed", _mode)
        strategy = STRATEGIES["fixed"]
    return strategy.chunk(pages, source)
