"""Local backend. The CV never leaves the machine, and iteration costs nothing."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from .base import LLMClient, LLMError, parse_json_response


class OllamaClient(LLMClient):
    def __init__(
        self,
        host: str,
        model: str,
        embed_model: str,
        timeout: float = 180.0,
        embed_query_prefix: str = "search_query: ",
        embed_document_prefix: str = "search_document: ",
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self.name = f"ollama:{model}"
        # Nomic's text embedding models (this project's documented default)
        # require these task prefixes for good retrieval quality; a different
        # embedding model that doesn't use this convention can blank them out.
        self.embed_query_prefix = embed_query_prefix
        self.embed_document_prefix = embed_document_prefix

    def complete_json(self, system: str, prompt: str, *, max_tokens: int = 8192) -> dict:
        try:
            resp = httpx.post(
                f"{self.host}/api/chat",
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    # Ollama can constrain decoding to valid JSON, which removes
                    # most of the malformed-output retries on small models.
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        return parse_json_response(resp.json().get("message", {}).get("content", ""))

    def embed(self, texts: Sequence[str], *, role: str = "document") -> list[list[float]]:
        prefix = self.embed_query_prefix if role == "query" else self.embed_document_prefix
        try:
            resp = httpx.post(
                f"{self.host}/api/embed",
                timeout=self.timeout,
                json={"model": self.embed_model, "input": [f"{prefix}{t}" for t in texts]},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama embed failed: {exc}") from exc
        vectors = resp.json().get("embeddings")
        if not vectors:
            raise LLMError("ollama returned no embeddings")
        return vectors
