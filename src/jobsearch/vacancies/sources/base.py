"""The port every vacancy source implements.

Four live sources disagree about description encoding, location shape and
timestamp format. All of that asymmetry is absorbed here so the rest of the
system sees one `Vacancy` type.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence

import httpx

from ..dedupe import vacancy_id
from ..models import Vacancy


class SourceError(RuntimeError):
    """A source failed. Ingest isolates this so other sources still finish."""


class VacancySource(abc.ABC):
    #: Stable identifier stored on every row and used by --source.
    name: str = ""

    @abc.abstractmethod
    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        """Return normalized vacancies, or raise SourceError."""

    def finalize(self, vacancies: Sequence[Vacancy]) -> list[Vacancy]:
        """Assign dedupe identities. Adapters never compute ids themselves."""
        for v in vacancies:
            v.id = vacancy_id(v.company, v.title, v.location_raw, v.description)
        return list(vacancies)


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    retries: int = 3,
):
    """GET with exponential backoff.

    A 404 is not retried: for the ATS sources it means a dead company slug, and
    retrying a slug that no longer exists just multiplies the wait.
    """
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                raise SourceError(f"404 Not Found: {url}")
            resp.raise_for_status()
            return resp.json()
        except SourceError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
    raise SourceError(f"{url}: {last}") from last


async def get_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    retries: int = 3,
):
    """GET a non-JSON body (Personio serves XML), with get_json's backoff.

    Redirects are not followed: for Personio, a dead tenant redirects to the
    vendor's marketing site instead of 404ing, so a redirect is the dead-slug
    signal here and fails fast the same way a 404 does in get_json.
    """
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await client.get(url, params=params, follow_redirects=False)
            if resp.status_code == 404 or resp.is_redirect:
                raise SourceError(f"{resp.status_code} (dead or moved): {url}")
            resp.raise_for_status()
            return resp.text
        except SourceError:
            raise
        except httpx.HTTPError as exc:
            last = exc
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
    raise SourceError(f"{url}: {last}") from last
