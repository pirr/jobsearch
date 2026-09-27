"""Ingest orchestration: fan out over sources, isolate their failures, store.

Sources are fetched concurrently but each one's outcome is independent. A
source that times out records its error in `ingest_runs` and the run continues
-- one dead company slug must never cost the other thirty-nine their results.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import yaml

from ..config import Config
from .sources.adzuna import AdzunaSource
from .sources.arbeitnow import ArbeitnowSource
from .sources.ats import ATS_SOURCES
from .sources.base import SourceError, VacancySource
from .sources.hackernews import HackerNewsSource
from .sources.himalayas import HimalayasSource
from .sources.jobicy import JobicySource
from .sources.remotejobsorg import RemoteJobsOrgSource
from .sources.remoteok import RemoteOkSource
from .sources.remotive import RemotiveSource
from .sources.workingnomads import WorkingNomadsSource
from .store import Store

USER_AGENT = "jobsearch-mvp/0.1 (+https://github.com/local/jobsearch)"

# Breadth sources: no company list needed, each covers many companies on its
# own. Depth (ATS) sources come from data/companies.yaml instead.
BREADTH_SOURCES: list[type[VacancySource]] = [
    ArbeitnowSource, RemoteOkSource, JobicySource, HimalayasSource, RemotiveSource,
    WorkingNomadsSource, RemoteJobsOrgSource, HackerNewsSource,
]


@dataclass
class SourceResult:
    label: str
    ok: bool
    fetched: int = 0
    new: int = 0
    updated: int = 0
    closed: int = 0
    error: str | None = None


def _label(source: VacancySource) -> str:
    return getattr(source, "label", source.name)


def build_sources(
    config: Config,
    only: str | None = None,
    company: str | None = None,
) -> list[VacancySource]:
    """Assemble the source list from config and the verified company file."""
    sources: list[VacancySource] = []

    if company:
        if not only or only not in ATS_SOURCES:
            raise ValueError("--company requires --source greenhouse|lever|ashby")
        return [ATS_SOURCES[only](company)]

    for cls in BREADTH_SOURCES:
        if only in (None, cls.name):
            sources.append(cls())

    # Adzuna is the only source needing credentials, so it can't sit in
    # BREADTH_SOURCES with the others (that loop calls `cls()` with no args).
    # Unconfigured means skipped silently, not an error, on every plain run;
    # explicitly requesting it via --source with no credentials set is still
    # a real error, same as an unknown source name.
    if config.adzuna_app_id and config.adzuna_app_key:
        if only in (None, AdzunaSource.name):
            sources.append(AdzunaSource(config.adzuna_app_id, config.adzuna_app_key))
    elif only == AdzunaSource.name:
        raise ValueError(
            "adzuna requires JOBSEARCH_ADZUNA_APP_ID and JOBSEARCH_ADZUNA_APP_KEY"
        )

    if config.companies_path.exists():
        listed = yaml.safe_load(config.companies_path.read_text()) or {}
        for platform, slugs in listed.items():
            if platform not in ATS_SOURCES:
                continue
            if only not in (None, platform):
                continue
            for slug in slugs or []:
                sources.append(ATS_SOURCES[platform](slug))
    elif only in ATS_SOURCES:
        raise ValueError(f"{config.companies_path} not found; nothing to ingest")

    if only and not sources:
        raise ValueError(f"unknown source {only!r}")
    return sources


async def _run_one(
    source: VacancySource,
    client: httpx.AsyncClient,
    store: Store,
    sem: asyncio.Semaphore,
    lock: asyncio.Lock,
) -> SourceResult:
    label = _label(source)
    run_id = store.start_run(label)
    async with sem:
        try:
            vacancies = await source.fetch(client)
        except SourceError as exc:
            store.finish_run(run_id, ok=False, count=0, error=str(exc))
            return SourceResult(label, ok=False, error=str(exc))
        except Exception as exc:  # a broken adapter must not kill the run
            msg = f"{type(exc).__name__}: {exc}"
            store.finish_run(run_id, ok=False, count=0, error=msg)
            return SourceResult(label, ok=False, error=msg)

    # sqlite3 connections are not safe to share across concurrent writers, so
    # writes are serialized while fetches stay parallel.
    async with lock:
        new, updated = store.upsert_vacancies(vacancies)
        closed = 0
        # Only company-scoped sources (Greenhouse/Lever/Ashby) fetch one
        # company's complete live listing per call, so only they can safely
        # say "anything stored here and missing from this fetch is gone" --
        # a breadth source (Arbeitnow/RemoteOK/Jobicy) covers many companies
        # per call and has no single company to diff against. An empty fetch
        # is skipped too: with no fetched vacancy to read a company name from,
        # there is no safe way to scope which stored rows even belong to it.
        company = getattr(source, "company", None)
        if vacancies and company is not None:
            closed = store.close_missing_vacancies(
                source.name, vacancies[0].company, {v.source_id for v in vacancies}
            )
        store.finish_run(run_id, ok=True, count=len(vacancies))
    return SourceResult(
        label, ok=True, fetched=len(vacancies), new=new, updated=updated, closed=closed
    )


async def ingest(
    config: Config,
    store: Store,
    sources: Sequence[VacancySource],
    progress=None,
) -> list[SourceResult]:
    sem = asyncio.Semaphore(config.max_concurrent_sources)
    lock = asyncio.Lock()
    results: list[SourceResult] = []
    async with httpx.AsyncClient(
        timeout=config.http_timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        tasks = [
            asyncio.create_task(_run_one(s, client, store, sem, lock)) for s in sources
        ]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if progress:
                progress(result)
    return results


async def check_sources(config: Config, sources: Sequence[VacancySource]) -> list[SourceResult]:
    """Health check for `doctor`: which slugs are alive, without storing anything."""
    sem = asyncio.Semaphore(config.max_concurrent_sources)
    results: list[SourceResult] = []

    async def one(source: VacancySource) -> SourceResult:
        async with sem:
            try:
                vacancies = await source.fetch(client)
                return SourceResult(_label(source), ok=True, fetched=len(vacancies))
            except Exception as exc:
                return SourceResult(_label(source), ok=False, error=f"{type(exc).__name__}: {exc}")

    async with httpx.AsyncClient(
        timeout=config.http_timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        results = list(await asyncio.gather(*(one(s) for s in sources)))
    return results
