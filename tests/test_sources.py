"""Adapters run against payloads recorded from the live APIs."""

import asyncio

import pytest
from conftest import FakeClient, fixture

from jobsearch.vacancies.dedupe import vacancy_id
from jobsearch.vacancies.ingest import _run_one
from jobsearch.vacancies.sources.arbeitnow import ArbeitnowSource
from jobsearch.vacancies.sources.ats import AshbySource, GreenhouseSource, LeverSource
from jobsearch.vacancies.sources.base import SourceError
from jobsearch.vacancies.sources.himalayas import HimalayasSource
from jobsearch.vacancies.sources.jobicy import JobicySource
from jobsearch.vacancies.sources.remoteok import RemoteOkSource
from jobsearch.vacancies.sources.remotive import RemotiveSource


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


async def test_remoteok_skips_the_legal_notice_row():
    # The feed's first element is an attribution notice, not a job -- it has
    # no "id" field, which is what the adapter filters on.
    vacancies = await RemoteOkSource().fetch(FakeClient(fixture("remoteok")))
    assert vacancies
    v = vacancies[0]
    assert v.source == "remoteok" and v.company and v.title and v.url
    assert v.is_remote is True
    assert "<" not in v.description[:200]


async def test_jobicy_requests_european_dev_roles():
    client = FakeClient(fixture("jobicy"))
    vacancies = await JobicySource().fetch(client)
    assert vacancies
    v = vacancies[0]
    assert v.source == "jobicy" and v.company and v.title and v.url
    assert v.is_remote is True
    assert "<" not in v.description[:200]
    _, params = client.requests[0]
    assert params["geo"] == "europe" and params["industry"] == "dev"


async def test_remotive_is_always_remote():
    vacancies = await RemotiveSource().fetch(FakeClient(fixture("remotive")))
    assert vacancies
    v = vacancies[0]
    assert v.source == "remotive" and v.company and v.title and v.url
    assert v.is_remote is True
    assert "<" not in v.description[:200]


async def test_himalayas_uses_guid_as_source_id_and_stops_without_cursor():
    # The fixture's nextCursor is None, so a single page must be fetched even
    # though max_pages defaults higher -- there is no numeric "id" field on a
    # Himalayas job, unlike every other source, so guid (a stable URL) is used.
    client = FakeClient(fixture("himalayas"))
    vacancies = await HimalayasSource().fetch(client)
    assert vacancies
    v = vacancies[0]
    assert v.source == "himalayas" and v.company and v.title
    assert v.source_id == v.raw["guid"]
    assert v.is_remote is True
    assert "<" not in v.description[:200]
    assert len(client.requests) == 1


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


def _greenhouse_payload(job_ids):
    return {"jobs": [
        {
            "id": jid, "title": f"Role {jid}", "company_name": "Acme",
            "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
            "location": {"name": "Berlin"}, "content": "<p>Do the work.</p>",
        }
        for jid in job_ids
    ]}


async def test_run_one_marks_a_vacancy_closed_when_a_later_fetch_omits_it(store):
    """The mechanism behind ingest's closed-vacancy detection: two successive
    fetches of the same company, the second missing a job the first had."""
    source = GreenhouseSource("acme")
    sem, lock = asyncio.Semaphore(1), asyncio.Lock()

    await _run_one(source, FakeClient(_greenhouse_payload([1, 2])), store, sem, lock)
    assert store.count_vacancies() == 2

    result = await _run_one(source, FakeClient(_greenhouse_payload([1])), store, sem, lock)

    assert result.closed == 1
    gone_id = vacancy_id("Acme", "Role 2", "Berlin", "Do the work.")
    stayed_id = vacancy_id("Acme", "Role 1", "Berlin", "Do the work.")
    store.save_rating(gone_id, "h", "f", score=80)
    store.save_rating(stayed_id, "h", "f", score=80)
    assert [r["title"] for r in store.ranked("h", 0, 10)] == ["Role 1"]
    assert store.ranked("h", 0, 10, only_status="closed")[0]["title"] == "Role 2"


async def test_run_one_skips_the_closed_check_for_breadth_sources(store):
    """Arbeitnow/RemoteOK/Jobicy cover many companies per call -- there is no
    single company to diff against, so the check must not run for them."""
    payload = fixture("arbeitnow")
    payload["links"]["next"] = None
    sem, lock = asyncio.Semaphore(1), asyncio.Lock()

    result = await _run_one(ArbeitnowSource(), FakeClient(payload), store, sem, lock)

    assert result.closed == 0
