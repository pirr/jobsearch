"""Local backend. The CV never leaves the machine, and iteration costs nothing."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from .base import LLMClient, LLMError, parse_json_response


class OllamaClient(LLMClient):
    def __init__(self, host: str, model: str, embed_model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self.name = f"ollama:{model}"

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

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                f"{self.host}/api/embed",
                timeout=self.timeout,
                json={"model": self.embed_model, "input": list(texts)},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama embed failed: {exc}") from exc
        vectors = resp.json().get("embeddings")
        if not vectors:
            raise LLMError("ollama returned no embeddings")
        return vectors
