"""Jobicy -- a third breadth source, scoped to European engineering roles.

No company list needed, one bounded request per ingest run. The API is
queried with `industry=dev` and `geo=europe` rather than pulled unfiltered:
Jobicy carries every remote category (sales, support, marketing, ...), and
this project only wants the slice relevant to a tech candidate's DACH/EU
search, matching Arbeitnow's regional focus.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://jobicy.com/api/v2/remote-jobs"


class JobicySource(VacancySource):
    name = "jobicy"

    def __init__(self, count: int = 100):
        self.count = count

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        params = {"count": self.count, "geo": "europe", "industry": "dev"}
        payload = await get_json(client, API, params=params)
        rows = payload.get("jobs") or []
        return self.finalize([self._to_vacancy(row) for row in rows])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, _ = normalize_location(row.get("jobGeo"))
        job_types = row.get("jobType") or []
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("url", ""),
            company=row.get("companyName", ""),
            title=row.get("jobTitle", ""),
            description=strip_html(row.get("jobDescription")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=job_types[0] if job_types else None,
            posted_at=normalize_timestamp(row.get("pubDate")),
            raw=row,
        )
