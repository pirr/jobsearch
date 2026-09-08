"""Adzuna -- a sixth breadth source, the only one that needs credentials.

Unlike every other source, Adzuna is a general job aggregator, not a
remote-jobs board: most of what it indexes is on-site. `what=remote` narrows
the query itself, but the response's `location` field is a geographic place
name ("Marlow, Buckinghamshire"), never the word "remote", so
`normalize_location`'s text-based remote inference will rarely fire here --
unlike Arbeitnow, this is expected, not a bug: an unconfirmed `is_remote`
falls back to the downstream regex checks in `matching/eligibility.py`.

Countries are hardcoded rather than configurable, same as Jobicy hardcoding
`geo=europe`. `ru` is deliberately excluded even though it may be granted on
the account -- this project never queries it. Belarus is not offered by
Adzuna's API at all, so no exclusion code is needed for it.

The `/jobs/{country}/search/{page}` path takes the page number in the URL,
not as a query param, unlike every other paginated source here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import VacancySource, get_json

API = "https://api.adzuna.com/v1/api/jobs"

# US/UK/EU markets only -- matches this project's brief. "ru" is granted on
# some accounts but is never queried; Belarus isn't an Adzuna market at all.
COUNTRIES = ["gb", "us", "de", "fr", "nl", "pl"]

RESULTS_PER_PAGE = 50  # Adzuna's documented max


class AdzunaSource(VacancySource):
    name = "adzuna"

    def __init__(
        self,
        app_id: str,
        app_key: str,
        countries: Sequence[str] = COUNTRIES,
        max_pages_per_country: int = 5,
        page_delay: float = 0.3,
    ):
        self.app_id = app_id
        self.app_key = app_key
        self.countries = countries
        self.max_pages_per_country = max_pages_per_country
        self.page_delay = page_delay

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        out: list[Vacancy] = []
        for country in self.countries:
            out.extend(await self._fetch_country(client, country))
        return self.finalize(out)

    async def _fetch_country(self, client: httpx.AsyncClient, country: str) -> list[Vacancy]:
        rows_out: list[Vacancy] = []
        for page in range(1, self.max_pages_per_country + 1):
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": RESULTS_PER_PAGE,
                "what": "remote",
                "content-type": "application/json",
            }
            payload = await get_json(client, f"{API}/{country}/search/{page}", params=params)
            rows = payload.get("results") or []
            if not rows:
                break
            rows_out.extend(self._to_vacancy(r, country) for r in rows)
            if len(rows) < RESULTS_PER_PAGE:
                break
            await asyncio.sleep(self.page_delay)
        return rows_out

    def _to_vacancy(self, row: dict, country: str) -> Vacancy:
        location = row.get("location") or {}
        raw_loc, city, _, is_remote = normalize_location(location.get("display_name"))
        return Vacancy(
            source=self.name,
            source_id=str(row.get("id", "")),
            url=row.get("redirect_url", ""),
            company=(row.get("company") or {}).get("display_name", ""),
            title=row.get("title", ""),
            description=strip_html(row.get("description")),
            location_raw=raw_loc,
            location_city=city,
            # The query is already scoped to one country's endpoint, so that
            # is a more reliable signal than guessing from UK county/city
            # names normalize_location's hints don't cover.
            location_country=country.upper(),
            is_remote=is_remote,
            employment_type=row.get("contract_type") or row.get("contract_time"),
            posted_at=normalize_timestamp(row.get("created")),
            raw=row,
        )
