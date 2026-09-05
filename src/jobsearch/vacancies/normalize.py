"""Turn each source's idiosyncratic payload into clean, comparable fields.

Every rule here exists because a live probe of the real API produced something
awkward -- see the design doc's source table.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# Block-level tags whose boundaries carry meaning: without this, list items and
# paragraphs run together into one unreadable line and the LLM loses structure.
_BLOCK_RE = re.compile(
    r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article)\b[^>]*>",
    re.IGNORECASE,
)


def strip_html(raw: str | None) -> str:
    """Plain text from HTML, including Arbeitnow's double-escaped variety.

    Arbeitnow returns `&lt;div class=&quot;...&quot;&gt;` -- HTML that has been
    entity-escaped a second time -- so a single unescape pass leaves live tags
    behind. Unescaping until it stops changing handles both that and the
    ordinary single-escaped case without special-casing the source.
    """
    if not raw:
        return ""
    text = raw
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


# Location strings arrive as "Munich", "Berlin", "Remote, US, Georgia",
# {"name": ...}. There is no country code anywhere, so normalization is
# necessarily lossy -- which is why inferred geography is a soft signal.
_REMOTE_RE = re.compile(r"\bremote\b|\bhome\s?office\b|\banywhere\b", re.IGNORECASE)
_HYBRID_RE = re.compile(r"\bhybrid\b|\bon-?site\b", re.IGNORECASE)

_COUNTRY_HINTS = {
    "germany": "DE", "deutschland": "DE", "berlin": "DE", "munich": "DE",
    "münchen": "DE", "hamburg": "DE", "cologne": "DE", "köln": "DE",
    "frankfurt": "DE", "stuttgart": "DE", "düsseldorf": "DE", "leipzig": "DE",
    "austria": "AT", "vienna": "AT", "wien": "AT",
    "switzerland": "CH", "zurich": "CH", "zürich": "CH", "geneva": "CH", "basel": "CH",
    "netherlands": "NL", "amsterdam": "NL", "rotterdam": "NL",
    "france": "FR", "paris": "FR",
    "spain": "ES", "madrid": "ES", "barcelona": "ES",
    "poland": "PL", "warsaw": "PL", "krakow": "PL",
    "uk": "GB", "united kingdom": "GB", "london": "GB", "england": "GB",
    "ireland": "IE", "dublin": "IE",
    "portugal": "PT", "lisbon": "PT",
    "italy": "IT", "milan": "IT", "rome": "IT",
    "usa": "US", "united states": "US", "us": "US", "new york": "US",
    "san francisco": "US", "california": "US", "texas": "US", "georgia": "US",
    "canada": "CA", "toronto": "CA",
}

_NOISE_TOKENS = {"remote", "hybrid", "on-site", "onsite", "flexible", "various", ""}

# Tokens that name a country/state rather than a city, so they may set the
# country but must never be mistaken for one.
_COUNTRY_ONLY = {
    "germany", "deutschland", "austria", "switzerland", "netherlands", "france",
    "spain", "poland", "uk", "united kingdom", "england", "ireland", "portugal",
    "italy", "usa", "united states", "us", "canada", "california", "texas",
    "georgia",
}


def normalize_location(raw: str | dict | None) -> tuple[str, str | None, str | None, bool | None]:
    """Return (raw_text, city, country_code, is_remote).

    city/country are best-effort: a miss yields None rather than a wrong guess,
    because a wrong country silently removes jobs the candidate could take.
    """
    if raw is None:
        return "", None, None, None
    if isinstance(raw, dict):
        raw = raw.get("name") or raw.get("location") or ""
    raw = str(raw).strip()
    if not raw:
        return "", None, None, None

    is_remote: bool | None = None
    if _REMOTE_RE.search(raw):
        is_remote = True
    elif _HYBRID_RE.search(raw):
        is_remote = False

    parts = [p.strip() for p in re.split(r"[,/|]|\s+-\s+", raw) if p.strip()]
    country: str | None = None
    city: str | None = None
    for part in parts:
        key = part.lower()
        if key in _NOISE_TOKENS:
            continue
        hit = _COUNTRY_HINTS.get(key)
        if hit is not None:
            if country is None:
                country = hit
            # A country/region token is never the city. "Remote, US, Georgia"
            # must not yield city="US".
            if key in _COUNTRY_ONLY:
                continue
        if city is None:
            # First meaningful segment is conventionally the city across all
            # four probed sources.
            city = part
    if city and city.lower() in _NOISE_TOKENS:
        city = None
    return raw, city, country, is_remote


def normalize_timestamp(value) -> datetime | None:
    """Accept unix epoch (Arbeitnow) or ISO-8601 (Greenhouse, Ashby)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
