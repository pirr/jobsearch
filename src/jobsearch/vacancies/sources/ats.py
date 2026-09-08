"""Greenhouse, Lever, Ashby and SmartRecruiters -- the depth sources.

All four are unauthenticated public endpoints returning full descriptions
(SmartRecruiters needs a second per-posting call for its; see below), but each
has a different response shape, and a company slug that worked last month may
404 today (probing found personio, contentful, getyourguide, blinkist and
sennder all dead). Slug rot surfaces through `jobsearch doctor` rather than
being swallowed as an empty result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json


class _CompanySource(VacancySource):
    """Shared shape: one company slug per call, no cross-company search."""

    def __init__(self, company: str, retries: int = 3):
        self.company = company
        # Lower for discover-ats: most slug guesses are dead, and the default
        # 3x exponential backoff exists for a company already known to be
        # real, not for bulk guessing against companies that mostly aren't.
        self.retries = retries

    @property
    def label(self) -> str:
        return f"{self.name}:{self.company}"


class GreenhouseSource(_CompanySource):
    name = "greenhouse"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://boards-api.greenhouse.io/v1/boards/{self.company}/jobs"
        payload = await get_json(
            client, url, params={"content": "true"}, retries=self.retries
        )
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
        payload = await get_json(
            client, url, params={"mode": "json"}, retries=self.retries
        )
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
        payload = await get_json(
            client, url, params={"includeCompensation": "true"}, retries=self.retries
        )
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


class SmartRecruitersSource(_CompanySource):
    """Unlike the other three, the list endpoint never inlines a description --
    a job's `jobAd.sections` only comes back from a second, per-posting GET
    (the list row's own `ref` field is that detail URL, so no URL template is
    duplicated here). A company with hundreds of open roles would otherwise
    turn into hundreds of requests every ingest run, so postings are capped at
    `max_postings` -- the same "fetch a bounded slice, let matching judge
    relevance" tradeoff Himalayas/Adzuna already make for their own
    pagination -- and detail calls are fanned out under their own semaphore so
    one company can't be hammered at unbounded concurrency.
    """

    name = "smartrecruiters"

    def __init__(
        self,
        company: str,
        retries: int = 3,
        max_postings: int = 150,
        detail_concurrency: int = 5,
    ):
        super().__init__(company, retries=retries)
        self.max_postings = max_postings
        self.detail_concurrency = detail_concurrency

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        rows = await self._list(client)
        sem = asyncio.Semaphore(self.detail_concurrency)

        async def one(row: dict) -> Vacancy:
            async with sem:
                return await self._to_vacancy(client, row)

        vacancies = await asyncio.gather(*(one(row) for row in rows))
        return self.finalize(list(vacancies))

    async def _list(self, client: httpx.AsyncClient) -> list[dict]:
        url = f"https://api.smartrecruiters.com/v1/companies/{self.company}/postings"
        page_size = 100
        rows: list[dict] = []
        offset = 0
        while offset < self.max_postings:
            limit = min(page_size, self.max_postings - offset)
            payload = await get_json(
                client, url, params={"limit": limit, "offset": offset}, retries=self.retries
            )
            content = payload.get("content") or []
            rows.extend(content)
            offset += len(content)
            if len(content) < limit or offset >= payload.get("totalFound", 0):
                break
        return rows

    async def _to_vacancy(self, client: httpx.AsyncClient, row: dict) -> Vacancy:
        # `ref` is the API's own detail URL for this posting -- reusing it
        # avoids hand-building the URL a second time.
        ref = row.get("ref")
        detail = await get_json(client, ref, retries=self.retries) if ref else {}
        sections = ((detail.get("jobAd") or {}).get("sections")) or {}
        description = "\n\n".join(
            strip_html(sections[key]["text"])
            for key in ("jobDescription", "qualifications", "additionalInformation", "companyDescription")
            if sections.get(key, {}).get("text")
        )
        location = row.get("location") or {}
        # Probing found a remote posting whose `city` held "Poland" (the
        # country, not a city) -- location.city is trusted as given rather
        # than second-guessed, same as every other source's best-effort
        # location handling; a wrong city is low-stakes here (display only).
        country = (location.get("country") or "").upper() or None
        employment = (row.get("typeOfEmployment") or {}).get("label")
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=detail.get("postingUrl") or ref or "",
            company=(row.get("company") or {}).get("name") or self.company,
            # SmartRecruiters returns titles with stray trailing whitespace
            # (e.g. "Data Operations Consultant ") -- strip() before it leaks
            # into `show` output or the LLM prompt.
            title=(row.get("name") or "").strip(),
            description=description,
            location_raw=location.get("fullLocation") or location.get("city") or "",
            location_city=location.get("city") or None,
            location_country=country,
            is_remote=location.get("remote"),
            employment_type=employment,
            posted_at=normalize_timestamp(row.get("releasedDate")),
            raw=row,
        )


ATS_SOURCES = {
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "ashby": AshbySource,
    "smartrecruiters": SmartRecruitersSource,
}
