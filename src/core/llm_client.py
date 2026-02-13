import logging
import time
from typing import Any

from openai import OpenAI

from src.core.config import MODEL, EMBEDDING_MODEL, OPENAI_TIMEOUT, MAX_API_RETRIES

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around OpenAI providing retried chat and embedding calls."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def _retry(self, operation: str, fn, *args, **kwargs) -> Any:
        """Execute a callable with exponential-backoff retry."""
        last_error: Exception | None = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_API_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(
                        "%s failed (attempt %d), retrying in %ds: %s",
                        operation, attempt + 1, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "%s failed after %d attempts: %s",
                        operation, MAX_API_RETRIES + 1, exc,
                    )
        raise last_error  # type: ignore[misc]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 500,
        model: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Call OpenAI chat completion with retry. Returns stripped response text."""
        _model = model or MODEL
        _timeout = timeout or OPENAI_TIMEOUT

        def _call():
            response = self._client.chat.completions.create(
                model=_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=_timeout,
            )
            return response.choices[0].message.content.strip()

        return self._retry("OpenAI chat", _call)

    def embed(self, texts: list[str], model: str | None = None) -> list[Any]:
        """Embed texts via OpenAI with retry. Returns list of embedding vectors."""
        _model = model or EMBEDDING_MODEL

        def _call():
            response = self._client.embeddings.create(
                model=_model,
                input=texts,
                timeout=OPENAI_TIMEOUT,
            )
            return [item.embedding for item in response.data]

        return self._retry("OpenAI embeddings", _call)
