"""Deterministic stand-in so matching logic is testable without a model."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .base import LLMClient


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, responses: list[dict] | None = None, dim: int = 32):
        self.responses = list(responses or [])
        self.dim = dim
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, prompt: str, *, max_tokens: int = 8192) -> dict:
        self.calls.append((system, prompt))
        if self.responses:
            return self.responses.pop(0)
        return {"score": 50, "hard_blockers": [], "matched_skills": [],
                "missing_skills": [], "summary_fit": "", "summary_gap": ""}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([digest[i % len(digest)] / 255.0 for i in range(self.dim)])
        return out
