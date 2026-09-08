"""Remotive -- a fourth breadth source, no company list needed.

The free feed is small (its docs push heavier users toward a paid API), so this
covers little volume on its own, but it costs nothing to include alongside the
other breadth sources. Every listing on Remotive is remote by definition, so
`is_remote` is not inferred from the location text the way it is for Arbeitnow.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://remotive.com/api/remote-jobs"


class RemotiveSource(VacancySource):
    name = "remotive"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        payload = await get_json(client, API)
        rows = payload.get("jobs") or []
        return self.finalize([self._to_vacancy(row) for row in rows])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, _ = normalize_location(row.get("candidate_required_location"))
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("url", ""),
            company=row.get("company_name", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=row.get("job_type") or None,
            posted_at=normalize_timestamp(row.get("publication_date")),
            raw=row,
        )
