"""Arbeitnow -- the breadth source: ~2,000 active EU/DACH jobs, no auth, no
company list to maintain.

Their terms ask that consumers not abuse the API and link back to the site, so
pagination is bounded and politely paced.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://www.arbeitnow.com/api/job-board-api"


class ArbeitnowSource(VacancySource):
    name = "arbeitnow"

    def __init__(self, max_pages: int = 25, page_delay: float = 0.3):
        self.max_pages = max_pages
        self.page_delay = page_delay

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        out: list[Vacancy] = []
        for page in range(1, self.max_pages + 1):
            payload = await get_json(client, API, params={"page": page})
            rows = payload.get("data") or []
            if not rows:
                break
            out.extend(self._to_vacancy(r) for r in rows)
            if not (payload.get("links") or {}).get("next"):
                break
            await asyncio.sleep(self.page_delay)
        return self.finalize(out)

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, loc_remote = normalize_location(row.get("location"))
        # The explicit `remote` boolean is authoritative; the location string is
        # only a fallback when the flag is absent.
        remote = row.get("remote")
        if remote is None:
            remote = loc_remote
        job_types = row.get("job_types") or []
        return Vacancy(
            source=self.name,
            source_id=str(row.get("slug", "")),
            url=row.get("url", ""),
            company=row.get("company_name", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=bool(remote) if remote is not None else None,
            employment_type=job_types[0] if job_types else None,
            posted_at=normalize_timestamp(row.get("created_at")),
            raw=row,
        )
