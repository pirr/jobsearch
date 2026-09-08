"""Himalayas -- a fifth breadth source, no company list needed.

The feed is huge (100k+ jobs total) and its documented `/jobs/api/search`
filter params (category, excluded locations, limit) turned out not to actually
filter anything when probed live, so this pulls the plain feed with bounded
cursor pagination instead -- the same "fetch a bounded slice, let matching
judge relevance" approach Arbeitnow uses for its page-based pagination.

There is no "id" field on a job row; `guid` is a stable per-job URL and is used
as the dedupe source_id instead. `pubDate`/`expiryDate` are unix epoch, unlike
Jobicy's ISO-8601 strings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://himalayas.app/jobs/api"
PAGE_SIZE = 20  # the API's confirmed max per page


class HimalayasSource(VacancySource):
    name = "himalayas"

    def __init__(self, max_pages: int = 25, page_delay: float = 0.5):
        self.max_pages = max_pages
        self.page_delay = page_delay

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        out: list[Vacancy] = []
        cursor: str | None = None
        for _ in range(self.max_pages):
            params = {"limit": PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            payload = await get_json(client, API, params=params)
            rows = payload.get("jobs") or []
            if not rows:
                break
            out.extend(self._to_vacancy(r) for r in rows)
            cursor = payload.get("nextCursor")
            if not cursor:
                break
            await asyncio.sleep(self.page_delay)
        return self.finalize(out)

    def _to_vacancy(self, row: dict) -> Vacancy:
        # Every listing on Himalayas is remote by definition; locationRestrictions
        # names eligible candidate countries, not a worksite, so it never carries
        # the word "remote" for normalize_location's inference to key off.
        locations = row.get("locationRestrictions") or []
        raw_loc, city, country, _ = normalize_location(", ".join(locations))
        return Vacancy(
            source=self.name,
            source_id=str(row.get("guid", "")),
            url=row.get("applicationLink", ""),
            company=row.get("companyName", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=row.get("employmentType") or None,
            posted_at=normalize_timestamp(row.get("pubDate")),
            raw=row,
        )
