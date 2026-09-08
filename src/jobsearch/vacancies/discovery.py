"""Reverse ATS discovery: given a company name, find its live job board.

Growing `data/companies.yaml` has so far meant hand-probing vendor APIs and
recording what's alive (see the file's own dated-batch comments). This module
automates the guessing: derive slug candidates from a company name, probe each
configured ATS platform for each candidate with no auth and no LLM tokens, and
report which combinations are live.

Only platforms whose `ATS_SOURCES` constructor takes a single slug string are
probeable -- a platform needing multi-field addressing (e.g. a future Workday
adapter, which needs tenant/site/shard) cannot be guessed this way and must
stay out of `PROBEABLE_ATS`.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import yaml

from .ingest import USER_AGENT
from .sources.ats import ATS_SOURCES
from .sources.base import SourceError

# Platforms discover-ats knows how to guess a single-string slug for. Kept as
# an explicit allowlist rather than iterating ATS_SOURCES.keys(), so a future
# platform with a differently-shaped constructor can be registered there
# without silently breaking (or being silently included in) probing here.
PROBEABLE_ATS: set[str] = {"greenhouse", "lever", "ashby", "smartrecruiters"}

_LEGAL_SUFFIXES = {
    "gmbh", "ag", "inc", "llc", "ltd", "corp", "co", "se", "sa", "bv", "group", "plc",
}
_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def slug_variants(name: str, max_variants: int = 8) -> list[str]:
    """Candidate ATS slugs for a company name.

    No diacritic transliteration -- "Käfer GmbH" only ever yields "käfer"-based
    variants, never "kaefer"/"kafer". That is a known, accepted gap: broadening
    it risks false-positive hits on an unrelated company more than it's worth.
    """
    words = [w for w in _SPLIT_RE.split(name.lower().strip()) if w]
    if not words:
        return []
    stripped = words[:-1] if len(words) > 1 and words[-1] in _LEGAL_SUFFIXES else words

    variants: list[str] = []
    for candidate_words in (stripped, words):
        for sep in ("-", "", "_"):
            variants.append(sep.join(candidate_words))
    return list(dict.fromkeys(v for v in variants if v))[:max_variants]


@dataclass
class ProbeResult:
    platform: str
    slug: str
    ok: bool
    fetched: int = 0
    error: str | None = None

    @property
    def hit(self) -> bool:
        return self.ok and self.fetched > 0


async def probe(
    client: httpx.AsyncClient, platform: str, slug: str, retries: int = 1
) -> ProbeResult:
    """Try one (platform, slug) combination. A 404/dead slug is expected, not
    an error worth surfacing -- most guesses come back this way by design."""
    source = ATS_SOURCES[platform](slug, retries=retries)
    try:
        vacancies = await source.fetch(client)
    except SourceError as exc:
        return ProbeResult(platform, slug, ok=False, error=str(exc))
    except Exception as exc:  # a transport hiccup must not kill the whole run
        return ProbeResult(platform, slug, ok=False, error=f"{type(exc).__name__}: {exc}")
    return ProbeResult(platform, slug, ok=True, fetched=len(vacancies))


async def discover(
    company_names: Sequence[str],
    platforms: Iterable[str] = PROBEABLE_ATS,
    max_variants: int = 8,
    concurrency: int = 8,
    retries: int = 1,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> list[ProbeResult]:
    """Probe every (name-variant, platform) combination and return all results.

    Concurrency defaults lower and retries faster-failing than ingest's own
    tuning: this traffic is bulk-guessing against companies not yet known to
    use these vendors, not fetching a pre-vetted company's listing, so it
    should stay conservative by default rather than inherit ingest's pacing.

    `client` is normally omitted (a short-lived AsyncClient is opened here);
    tests pass a fake in directly, the same seam `_run_one` exposes in
    `ingest.py`.
    """
    wanted = [p for p in platforms if p in PROBEABLE_ATS]
    sem = asyncio.Semaphore(concurrency)

    async def bound(client: httpx.AsyncClient, platform: str, slug: str) -> ProbeResult:
        async with sem:
            return await probe(client, platform, slug, retries=retries)

    async def run(client: httpx.AsyncClient) -> list[ProbeResult]:
        tasks = [
            bound(client, platform, slug)
            for name in company_names
            for slug in slug_variants(name, max_variants=max_variants)
            for platform in wanted
        ]
        return list(await asyncio.gather(*tasks)) if tasks else []

    if client is not None:
        return await run(client)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as owned_client:
        return await run(owned_client)


def hits(results: Sequence[ProbeResult]) -> list[ProbeResult]:
    return [r for r in results if r.hit]


def write_discoveries(
    companies_path: Path, found: Sequence[ProbeResult], note: str | None = None
) -> dict[str, list[str]]:
    """Append newly-discovered slugs to companies.yaml, grouped by platform.

    This is not a general YAML editor -- it only ever appends scalar list
    items, in the file's own dated-batch-comment style, and never rewrites an
    existing line. Slugs already present (exact string match -- some slugs in
    this file are deliberately case-sensitive) are skipped. The write is
    atomic (temp file + rename) and self-validated by re-parsing the result;
    a parse failure aborts before anything touches the real file.

    Returns {platform: [slug, ...]} for what was actually written, so the
    caller can report it -- an empty dict means every hit was already known.
    """
    note = note or f"discovered via `jobsearch discover-ats` {date.today().isoformat()}"
    text = companies_path.read_text(encoding="utf-8") if companies_path.exists() else ""
    existing = (yaml.safe_load(text) or {}) if text.strip() else {}

    by_platform: dict[str, list[str]] = {}
    for r in found:
        already = existing.get(r.platform) or []
        pending = by_platform.setdefault(r.platform, [])
        if r.slug not in already and r.slug not in pending:
            pending.append(r.slug)
    by_platform = {p: slugs for p, slugs in by_platform.items() if slugs}
    if not by_platform:
        return {}

    for platform, slugs in by_platform.items():
        insertion = "".join(f"  # {note}\n  - {slug}\n" for slug in slugs)
        block_re = re.compile(rf"^{re.escape(platform)}:[ \t]*$", re.MULTILINE)
        match = block_re.search(text)
        if match is None:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"\n{platform}:\n{insertion}"
            continue
        rest = text[match.end():]
        next_section = re.search(r"^\S", rest, re.MULTILINE)
        insert_at = match.end() + (next_section.start() if next_section else len(rest))
        text = text[:insert_at] + insertion + text[insert_at:]

    tmp_path = companies_path.with_name(companies_path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    reparsed = yaml.safe_load(tmp_path.read_text(encoding="utf-8")) or {}
    ok = all(
        set(slugs) <= set(reparsed.get(platform) or []) for platform, slugs in by_platform.items()
    )
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"discovery write would have produced invalid YAML; {companies_path} left untouched"
        )
    os.replace(tmp_path, companies_path)
    return by_platform
