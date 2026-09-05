"""RemoteOK -- a second breadth source, no company list needed.

The feed's first element is a legal/attribution notice, not a job; unlike
every real posting it carries no "id" field, which is what separates it out.
Every listing on RemoteOK is remote by definition, so `is_remote` is not
inferred from the location text the way it is for the other sources.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://remoteok.com/api"


class RemoteOkSource(VacancySource):
    name = "remoteok"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        payload = await get_json(client, API)
        rows = [row for row in payload if isinstance(row, dict) and "id" in row]
        return self.finalize([self._to_vacancy(row) for row in rows])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, _ = normalize_location(row.get("location"))
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("url") or row.get("apply_url", ""),
            company=row.get("company", ""),
            title=row.get("position", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=None,  # not exposed as a typed field
            posted_at=normalize_timestamp(row.get("epoch")),
            raw=row,
        )
