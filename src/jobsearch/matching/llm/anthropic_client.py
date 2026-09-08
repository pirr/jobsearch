"""Cloud backend.

Sending a CV here means sending personal data to a third party -- a deliberate
choice, not a default. `JOBSEARCH_LLM=anthropic` selects it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import LLMClient, LLMError, parse_json_response


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "anthropic package not installed -- pip install 'jobsearch[cloud]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def complete_json(self, system: str, prompt: str, *, max_tokens: int = 8192) -> dict:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        return parse_json_response(text)

    def supports_embeddings(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, role: str = "document") -> list[list[float]]:
        raise LLMError(
            "Anthropic offers no embedding endpoint. Use JOBSEARCH_RETRIEVER=bm25, "
            "or keep Ollama available for embeddings."
        )
