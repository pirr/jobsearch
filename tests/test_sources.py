"""Adapters run against payloads recorded from the live APIs."""

import pytest
from conftest import FakeClient, fixture

from jobsearch.vacancies.sources.arbeitnow import ArbeitnowSource
from jobsearch.vacancies.sources.ats import AshbySource, GreenhouseSource, LeverSource
from jobsearch.vacancies.sources.base import SourceError


async def test_arbeitnow_normalizes_double_escaped_html():
    payload = fixture("arbeitnow")
    payload["links"]["next"] = None  # one page only
    vacancies = await ArbeitnowSource().fetch(FakeClient(payload))
    assert vacancies
    v = vacancies[0]
    assert "&lt;" not in v.description and "<div" not in v.description
    assert v.source == "arbeitnow" and v.company and v.title and v.url
    assert v.posted_at is not None       # unix epoch parsed
    assert all(x.id for x in vacancies)  # dedupe identity assigned


async def test_greenhouse_unwraps_location_object():
    vacancies = await GreenhouseSource("celonis").fetch(FakeClient(fixture("greenhouse")))
    assert vacancies
    v = vacancies[0]
    assert isinstance(v.location_raw, str)   # arrived as {"name": ...}
    assert "<p>" not in v.description
    assert v.url.startswith("http")


async def test_ashby_keeps_plain_text_unmangled():
    vacancies = await AshbySource("deepl").fetch(FakeClient(fixture("ashby")))
    assert vacancies
    v = vacancies[0]
    assert len(v.description) > 500
    assert "<" not in v.description[:200]


async def test_lever_concatenates_section_blocks():
    payload = fixture("lever")
    vacancies = await LeverSource("ro").fetch(FakeClient(payload))
    assert vacancies
    # Lever splits the body across `lists`; taking only `description` would
    # hand the rater the intro paragraph and nothing else.
    assert len(vacancies[0].description) > len(payload[0].get("descriptionPlain") or "")


async def test_dead_slug_raises_rather_than_returning_empty():
    # A 404 means the company slug is dead. Silently ingesting nothing would
    # hide that; `doctor` depends on this surfacing.
    with pytest.raises(SourceError):
        await GreenhouseSource("gone").fetch(FakeClient({}, status_code=404))


async def test_404_is_not_retried():
    client = FakeClient({}, status_code=404)
    with pytest.raises(SourceError):
        await AshbySource("gone").fetch(client)
    assert len(client.requests) == 1
