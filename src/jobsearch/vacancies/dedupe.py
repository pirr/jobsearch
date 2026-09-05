"""Canonical identity for a vacancy.

The same role is routinely posted to several boards and reposted weekly by the
same employer. Without a stable identity the ranked list fills with duplicates
of whichever match scores best, and re-ingesting re-pays for LLM ratings the
system already has.
"""

from __future__ import annotations

import hashlib
import re

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Decoration employers add to the same underlying role. Stripping it keeps
# "Senior Backend Engineer (m/f/d)" and "Senior Backend Engineer" one vacancy.
_TITLE_NOISE_RE = re.compile(
    r"\(\s*(m\s*/\s*f\s*/\s*[dx]|w\s*/\s*m\s*/\s*[dx]|all genders?|remote|hybrid)\s*\)",
    re.IGNORECASE,
)


def _canon(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = _TITLE_NOISE_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def vacancy_id(company: str, title: str, location: str, description: str) -> str:
    """Stable hash over the fields that identify a role.

    Description is truncated to its first 500 characters: enough to separate two
    genuinely different roles with the same title at one company, short enough
    that an edited paragraph further down does not fork the identity and
    resurrect a vacancy the candidate already dismissed.
    """
    payload = "␟".join(
        (_canon(company), _canon(title), _canon(location), _canon(description)[:500])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
