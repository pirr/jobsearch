"""Greenhouse, Lever and Ashby -- the depth sources.

All three are unauthenticated public endpoints returning full descriptions, but
each has a different response shape, and a company slug that worked last month
may 404 today (probing found personio, contentful, getyourguide, blinkist and
sennder all dead). Slug rot surfaces through `jobsearch doctor` rather than
being swallowed as an empty result.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json


class _CompanySource(VacancySource):
    """Shared shape: one company slug per call, no cross-company search."""

    def __init__(self, company: str):
        self.company = company

    @property
    def label(self) -> str:
        return f"{self.name}:{self.company}"


class GreenhouseSource(_CompanySource):
    name = "greenhouse"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://boards-api.greenhouse.io/v1/boards/{self.company}/jobs"
        payload = await get_json(client, url, params={"content": "true"})
        return self.finalize([self._to_vacancy(j) for j in payload.get("jobs", [])])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, remote = normalize_location(row.get("location"))
        # Greenhouse hides employment type in a freeform metadata list rather
        # than a typed field.
        employment = None
        for meta in row.get("metadata") or []:
            if str(meta.get("name", "")).lower() in {"employment type", "job type"}:
                value = meta.get("value")
                employment = value if isinstance(value, str) else None
                break
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("absolute_url", ""),
            company=row.get("company_name") or self.company,
            title=row.get("title", ""),
            description=strip_html(row.get("content")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=remote,
            employment_type=employment,
            posted_at=normalize_timestamp(
                row.get("first_published") or row.get("updated_at")
            ),
            raw=row,
        )


class LeverSource(_CompanySource):
    name = "lever"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://api.lever.co/v0/postings/{self.company}"
        payload = await get_json(client, url, params={"mode": "json"})
        if not isinstance(payload, list):
            return []
        return self.finalize([self._to_vacancy(j) for j in payload])

    def _to_vacancy(self, row: dict) -> Vacancy:
        cats = row.get("categories") or {}
        raw_loc, city, country, remote = normalize_location(cats.get("location"))
        # Lever splits the body across description + a list of section blocks;
        # rating needs all of it, not just the intro.
        parts = [row.get("descriptionPlain") or row.get("description") or ""]
        for section in row.get("lists") or []:
            parts.append(section.get("text", ""))
            parts.append(section.get("content", ""))
        parts.append(row.get("additionalPlain") or row.get("additional") or "")
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("hostedUrl") or row.get("applyUrl", ""),
            company=self.company,
            title=row.get("text", ""),
            description=strip_html("\n\n".join(p for p in parts if p)),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=remote,
            employment_type=cats.get("commitment"),
            posted_at=normalize_timestamp(
                row.get("createdAt") // 1000 if isinstance(row.get("createdAt"), int)
                else row.get("createdAt")
            ),
            raw=row,
        )


class AshbySource(_CompanySource):
    name = "ashby"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://api.ashbyhq.com/posting-api/job-board/{self.company}"
        payload = await get_json(client, url, params={"includeCompensation": "true"})
        jobs = [j for j in payload.get("jobs", []) if j.get("isListed", True)]
        return self.finalize([self._to_vacancy(j) for j in jobs])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, loc_remote = normalize_location(row.get("location"))
        remote = row.get("isRemote")
        if remote is None:
            remote = loc_remote
        # Ashby already returns plain text -- running it through strip_html
        # would be lossy for no gain.
        description = row.get("descriptionPlain") or strip_html(row.get("descriptionHtml"))
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("jobUrl") or row.get("applyUrl", ""),
            company=self.company,
            title=row.get("title", ""),
            description=description.strip(),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=bool(remote) if remote is not None else None,
            employment_type=row.get("employmentType"),
            posted_at=normalize_timestamp(row.get("publishedAt")),
            raw=row,
        )


ATS_SOURCES = {
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "ashby": AshbySource,
}
