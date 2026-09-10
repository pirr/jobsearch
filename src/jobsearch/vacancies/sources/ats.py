"""Greenhouse, Lever, Ashby, SmartRecruiters, Workable, Recruitee and
Teamtailor -- the depth sources.

All seven are unauthenticated public endpoints returning full descriptions
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
    """Unlike the other six, the list endpoint never inlines a description --
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


class WorkableSource(_CompanySource):
    name = "workable"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://apply.workable.com/api/v1/widget/accounts/{self.company}"
        payload = await get_json(
            client, url, params={"details": "true"}, retries=self.retries
        )
        # The wrapper has no per-job company field, only this one account-wide
        # display name -- "Hugging Face", not the "huggingface" slug.
        company = payload.get("name") or self.company
        return self.finalize(
            [self._to_vacancy(j, company) for j in payload.get("jobs", [])]
        )

    def _to_vacancy(self, row: dict, company: str) -> Vacancy:
        # Probing found neither a numeric "id" nor a separate "full_description"
        # field the plan expected -- `details=true` fills the full text into
        # "description" itself, and "shortcode" is the only identifier. The
        # per-job "locations" entry already carries an ISO-2 `countryCode`
        # though, so unlike Greenhouse/Ashby's freeform strings, location here
        # doesn't need normalize_location's city/country guessing.
        loc = (row.get("locations") or [{}])[0] or {}
        city = loc.get("city") or row.get("city") or None
        raw_loc = ", ".join(
            p for p in (city, loc.get("region"), loc.get("country") or row.get("country")) if p
        )
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id") or row.get("shortcode") or ""),
            url=row.get("url") or row.get("shortlink") or "",
            company=company,
            title=row.get("title", ""),
            description=strip_html(row.get("full_description") or row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=loc.get("countryCode") or None,
            is_remote=row.get("telecommuting"),
            employment_type=row.get("employment_type"),
            posted_at=normalize_timestamp(row.get("published_on") or row.get("created_at")),
            raw=row,
        )


class RecruiteeSource(_CompanySource):
    name = "recruitee"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://{self.company}.recruitee.com/api/offers/"
        payload = await get_json(client, url, retries=self.retries)
        return self.finalize([self._to_vacancy(o) for o in payload.get("offers", [])])

    def _to_vacancy(self, row: dict) -> Vacancy:
        # Unlike SmartRecruiters, the full HTML description already rides in
        # this list response -- no second per-posting call needed.
        #
        # `published_at` arrives as "2026-08-14 08:23:10 UTC": a space instead
        # of "T" and a trailing zone name instead of an offset, which
        # normalize_timestamp's datetime.fromisoformat can't parse (it fails
        # silently and returns None). Rewritten into ISO-8601 here, the same
        # spirit as Lever's own createdAt // 1000 conversion above, rather
        # than teaching the shared normalizer one source's timestamp dialect.
        published_at = row.get("published_at")
        if isinstance(published_at, str) and published_at.endswith(" UTC"):
            published_at = published_at.replace(" UTC", "").replace(" ", "T") + "+00:00"
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("careers_url", ""),
            company=row.get("company_name") or self.company,
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=row.get("location") or row.get("city") or "",
            location_city=row.get("city") or None,
            location_country=row.get("country_code") or None,
            is_remote=row.get("remote"),
            employment_type=row.get("employment_type_code"),
            posted_at=normalize_timestamp(published_at),
            raw=row,
        )


class TeamtailorSource(_CompanySource):
    """`{tenant}.teamtailor.com/jobs.json` is a JSON Feed 1.1 (`content-type:
    application/feed+json`), not a bespoke schema -- verified live against six
    real tenants (career/recruitgo/oneractive/viviennewestwood/proxima/
    inforcer.teamtailor.com) on 2026-09-10. The full HTML job body already
    rides inline as `content_html` (thousands of characters, not a teaser),
    same shape as Recruitee -- no second per-posting call needed. Each item
    also carries a schema.org `_jobposting` block that the plain feed fields
    don't have on their own (structured location, `hiringOrganization`); none
    of the six tenants ever populated `employmentType` on it, so that mapping
    is best-effort/likely-always-None rather than proven live.
    """

    name = "teamtailor"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        url = f"https://{self.company}.teamtailor.com/jobs.json"
        payload = await get_json(client, url, retries=self.retries)
        return self.finalize([self._to_vacancy(item) for item in payload.get("items", [])])

    def _to_vacancy(self, row: dict) -> Vacancy:
        posting = row.get("_jobposting") or {}
        # A posting can list several office locations at once (real proxima
        # postings span 3-4 UK cities) -- unlike every single-location ATS
        # above, city/country come from the first entry only, but the raw
        # string joins every distinct city (de-duplicated: one live posting
        # repeated "Cardiff" twice with a slightly different address) rather
        # than silently dropping the rest.
        places = [loc.get("address") or {} for loc in (posting.get("jobLocation") or [])]
        cities = list(
            dict.fromkeys(p.get("addressLocality") for p in places if p.get("addressLocality"))
        )
        first = places[0] if places else {}
        organization = posting.get("hiringOrganization") or {}
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("url", ""),
            company=organization.get("name") or self.company,
            title=row.get("title", ""),
            # `_jobposting.description` duplicates `content_html` verbatim on
            # every tenant probed -- content_html is the feed's own field, so
            # it's used directly rather than reaching into the schema.org copy.
            description=strip_html(row.get("content_html")),
            location_raw=", ".join(cities),
            location_city=first.get("addressLocality") or None,
            location_country=(first.get("addressCountry") or "").upper() or None,
            # No live posting across any of the six tenants surfaced a remote
            # signal (no "Remote" locality, no dedicated flag) -- unlike
            # Workable/SmartRecruiters, this is a real gap, not an unmapped
            # field: None reflects that honestly rather than guessing from
            # description text the way normalize_location does for a string.
            is_remote=None,
            employment_type=posting.get("employmentType"),
            # Unlike Recruitee's " UTC"-suffixed timestamp, date_published
            # arrives as ordinary offset ISO-8601 ("+02:00" or "Z") --
            # normalize_timestamp's fromisoformat parses it with no rewrite.
            posted_at=normalize_timestamp(row.get("date_published")),
            raw=row,
        )


ATS_SOURCES = {
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "ashby": AshbySource,
    "smartrecruiters": SmartRecruitersSource,
    "workable": WorkableSource,
    "recruitee": RecruiteeSource,
    "teamtailor": TeamtailorSource,
}
