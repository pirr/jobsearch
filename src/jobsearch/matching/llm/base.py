"""The LLM port.

Local and cloud sit behind one interface so the same golden set can score both
and the choice is made on evidence. Embeddings live here too: the retriever
needs vectors, and routing them through this port avoids adding a second heavy
model dependency alongside the one that already exists.
"""

from __future__ import annotations

import abc
import json
import re
from collections.abc import Sequence


class LLMError(RuntimeError):
    pass


class LLMClient(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def complete_json(self, system: str, prompt: str, *, max_tokens: int = 8192) -> dict:
        """Return parsed JSON. Raise LLMError if the model won't produce any."""

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embedding vectors, one per input text."""

    def supports_embeddings(self) -> bool:
        return True


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _iter_balanced_objects(text: str):
    """Yield (start, end) spans of every top-level {...} object in text.

    A model that has no separate reasoning field (unlike one that exposes
    `reasoning_content`) writes its chain of thought directly into the same
    text as the final answer. A naive `text.find("{")...text.rfind("}")`
    breaks the moment that reasoning mentions any brace character before the
    real JSON -- observed directly: one response failed to parse at all, and
    an identical retry succeeded, which pointed at exactly this fragility.

    This walks the text once, tracking JSON string boundaries (so a brace
    inside a quoted value -- a job posting quoting a template literal like
    `${var}`, say -- is never mistaken for structural nesting) and brace
    depth, yielding each complete top-level object as it closes.
    """
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield start, i + 1


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response.

    Smaller local models habitually wrap JSON in prose or code fences even when
    told not to, so the strict parse is tried first and the salvage paths only
    run when it fails.
    """
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _FENCE_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Largest first, not last first: when reasoning and the answer share one
    # text, a stray syntactically-valid fragment can appear either BEFORE the
    # real answer (mentioned in passing while reasoning) or AFTER it (a
    # repeated/duplicated tail -- a known degenerate-generation pattern,
    # observed directly: one real response ended with a single leftover
    # matched-skill dict repeated after the full, correct answer had already
    # closed). Position alone can't distinguish those two cases, but our
    # schemas are always richer than a stray fragment, so size is the robust
    # signal. Equal-size ties fall back to preferring the later span.
    candidates = list(_iter_balanced_objects(text))
    candidates.sort(key=lambda span: (span[1] - span[0], span[0]), reverse=True)
    for start, end in candidates:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    raise LLMError(f"no JSON object in response: {text[:200]!r}")
