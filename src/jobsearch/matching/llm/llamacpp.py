"""llama.cpp's own server (llama-server), not Ollama.

It speaks the OpenAI-compatible API (/v1/chat/completions, /v1/embeddings)
rather than Ollama's native one, so this is a separate adapter behind the same
port -- the request and response shapes genuinely differ.

Behind llama-swap (a common front for llama.cpp that multiplexes several model
configs), the `model` field is not decorative -- it is how the proxy routes
the request, and chat and embeddings are always different models. That is why
this client, like OllamaClient, takes two model names rather than one.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from .base import LLMClient, LLMError, parse_json_response


def _http_error_detail(exc: httpx.HTTPError) -> str:
    """The server's own error body, when there is one.

    A bare str(exc) on an HTTPStatusError is just "400 Bad Request for url
    ...", which threw away the one thing worth knowing -- e.g. llama.cpp's
    "input (3167 tokens) is larger than the max context size (2048 tokens)".
    Surfacing the body is what makes failures like that diagnosable without
    reproducing the request by hand.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        body = response.text.strip()
        if body:
            return f"{exc} -- {body[:400]}"
    return str(exc)


class LlamaCppClient(LLMClient):
    def __init__(
        self,
        host: str,
        model: str = "local",
        embed_model: str | None = None,
        timeout: float = 180.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model or model
        self.timeout = timeout
        self.name = f"llamacpp:{model}"

    def complete_json(self, system: str, prompt: str, *, max_tokens: int = 8192) -> dict:
        try:
            resp = httpx.post(
                f"{self.host}/v1/chat/completions",
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    # Plain JSON-object mode, not llama.cpp's json_schema mode --
                    # the latter has open bugs when a grammar is also active,
                    # and the salvage parser below already tolerates a model
                    # that ignores this hint and wraps output in prose.
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"llama.cpp chat request failed for model {self.model!r}: {_http_error_detail(exc)}"
            ) from exc
        try:
            choice = resp.json()["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected llama.cpp response shape: {exc}") from exc
        if not content.strip() and choice.get("finish_reason") == "length":
            # A reasoning model (Qwen3-family and similar) writes its chain of
            # thought to a separate `reasoning_content` field. If it exhausts
            # max_tokens while still reasoning, `content` never gets written
            # at all -- this is that failure, not a malformed response, and
            # the fix is a bigger max_tokens, not a retry.
            reasoning_len = len(choice.get("message", {}).get("reasoning_content") or "")
            raise LLMError(
                f"model {self.model!r} used the full max_tokens={max_tokens} budget on "
                f"reasoning ({reasoning_len} chars of reasoning_content) and never wrote "
                "an answer -- pass a larger max_tokens, or disable the model's reasoning "
                "mode if it supports that"
            )
        return parse_json_response(content)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                f"{self.host}/v1/embeddings",
                timeout=self.timeout,
                json={"model": self.embed_model, "input": list(texts)},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"llama.cpp embeddings request failed for model {self.embed_model!r}: "
                f"{_http_error_detail(exc)} -- if this is a context-size error, the "
                "retriever's per-document character cap is too generous for this "
                "model's context window; a bare llama-server must also be started "
                "with --embedding for this endpoint to work"
            ) from exc
        data = sorted(resp.json().get("data", []), key=lambda d: d.get("index", 0))
        if not data:
            raise LLMError("llama.cpp returned no embeddings")
        return [d["embedding"] for d in data]
