"""RemoteJobs.org -- a seventh breadth source, no company list needed.

The response is paginated (`pagination.total` runs into the thousands), but
this only ever fetches the first page: no loop is built for it, the same
"fetch a bounded slice, let matching judge relevance" tradeoff already
accepted elsewhere (Himalayas/Adzuna/SmartRecruiters) rather than adding
pagination complexity to a breadth source that re-ingests periodically anyway.
`meta.powered_by` in the live response is the API's own attribution string --
it lives on the top-level payload, not any individual row, so it never reaches
a Vacancy's `raw`; nothing here strips or duplicates it, and no extra
attribution string needs adding on our side. Every listing is
inherently remote (it's a remote-jobs board), so `is_remote` is set directly
rather than inferred from the location text -- which is sometimes a country
("India") rather than "Remote", unlike the other remote-only breadth sources.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://remotejobs.org/api/v1/jobs"


class RemoteJobsOrgSource(VacancySource):
    name = "remotejobsorg"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        payload = await get_json(client, API)
        rows = payload.get("data") or []
        return self.finalize([self._to_vacancy(row) for row in rows])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, _ = normalize_location(row.get("location"))
        company = row.get("company") or {}
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("url") or row.get("apply_url", ""),
            company=company.get("name", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=row.get("type") or None,
            posted_at=normalize_timestamp(row.get("posted_at")),
            raw=row,
        )
