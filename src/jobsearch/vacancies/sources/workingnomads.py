"""Working Nomads -- a sixth breadth source, no company list needed.

The feed carries no "id" field at all (unlike RemoteOK/Remotive/Jobicy), so
the job's own URL -- already unique per posting -- doubles as source_id, the
same fallback Himalayas uses for its guid. Every listing here is inherently
remote (it's a remote-jobs board), so `is_remote` is set directly rather than
inferred from the location text, which is often a timezone/eligibility note
("CET (+/- 3 hours)", "USA, Canada or UK only") rather than a place.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://www.workingnomads.com/api/exposed_jobs/"


class WorkingNomadsSource(VacancySource):
    name = "workingnomads"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        payload = await get_json(client, API)
        return self.finalize([self._to_vacancy(row) for row in payload])

    def _to_vacancy(self, row: dict) -> Vacancy:
        raw_loc, city, country, _ = normalize_location(row.get("location"))
        url = row.get("url", "")
        return Vacancy(
            source=self.name,
            source_id=url,
            url=url,
            company=row.get("company_name", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            location_country=country,
            is_remote=True,
            employment_type=None,  # not exposed as a typed field
            posted_at=normalize_timestamp(row.get("pub_date")),
            raw=row,
        )
