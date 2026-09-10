"""Adapters run against payloads recorded from the live APIs."""

import asyncio

import pytest
from conftest import FakeClient, fixture

from jobsearch.config import Config
from jobsearch.vacancies.dedupe import vacancy_id
from jobsearch.vacancies.ingest import _run_one, build_sources
from jobsearch.vacancies.sources.adzuna import AdzunaSource
from jobsearch.vacancies.sources.arbeitnow import ArbeitnowSource
from jobsearch.vacancies.sources.ats import (
    AshbySource,
    GreenhouseSource,
    LeverSource,
    RecruiteeSource,
    SmartRecruitersSource,
    TeamtailorSource,
    WorkableSource,
)
from jobsearch.vacancies.sources.base import SourceError
from jobsearch.vacancies.sources.himalayas import HimalayasSource
from jobsearch.vacancies.sources.jobicy import JobicySource
from jobsearch.vacancies.sources.remotejobsorg import RemoteJobsOrgSource
from jobsearch.vacancies.sources.remoteok import RemoteOkSource
from jobsearch.vacancies.sources.remotive import RemotiveSource
from jobsearch.vacancies.sources.workingnomads import WorkingNomadsSource


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


async def test_smartrecruiters_fetches_description_via_a_second_detail_call():
    fx = fixture("smartrecruiters")

    def payload(call_count):
        return fx["list"] if call_count == 1 else fx["detail"]

    client = FakeClient(payload)
    vacancies = await SmartRecruitersSource("smartrecruiters").fetch(client)
    assert len(client.requests) == 2  # one list call, one detail call
    assert vacancies
    v = vacancies[0]
    assert v.source == "smartrecruiters" and v.company == "SmartRecruiters Inc"
    # The live payload has a trailing space in "name": "Data Operations Consultant "
    assert v.title == "Data Operations Consultant"
    assert v.url.startswith("https://jobs.smartrecruiters.com/")
    assert v.is_remote is True
    assert v.location_country == "PL"
    assert "<" not in v.description and len(v.description) > 100


async def test_workable_reads_shortcode_and_nested_country_code():
    fx = fixture("workable")
    vacancies = await WorkableSource("huggingface").fetch(FakeClient(fx))
    assert vacancies
    v = vacancies[0]
    assert v.source == "workable"
    # There's no numeric "id" on this endpoint -- "shortcode" is the identifier.
    assert v.source_id == fx["jobs"][0]["shortcode"]
    # The account-wide "name" is used, not the "huggingface" slug.
    assert v.company == "Hugging Face"
    # The nested locations[0] entry already carries an ISO countryCode.
    assert v.location_country == "FR"
    assert v.location_city == "Paris"
    assert v.is_remote is True
    assert "<" not in v.description[:200]
    assert v.url.startswith("https://apply.workable.com/")


async def test_recruitee_description_needs_no_second_call():
    client = FakeClient(fixture("recruitee"))
    vacancies = await RecruiteeSource("bunq").fetch(client)
    assert len(client.requests) == 1  # unlike SmartRecruiters, one call is enough
    assert len(vacancies) == 2
    v = vacancies[0]
    assert v.source == "recruitee" and v.company == "bunq"
    assert "<" not in v.description and len(v.description) > 100
    assert v.location_country == "NL"
    assert v.is_remote is False   # on_site=True, remote=False
    # The second posting is hybrid=True but still remote=False -- is_remote
    # tracks only the dedicated "remote" flag, not "hybrid" too.
    assert vacancies[1].is_remote is False


async def test_recruitee_published_at_survives_the_non_iso_timestamp():
    # published_at arrives as "2026-09-07 12:05:26 UTC" -- a space instead of
    # "T" and a trailing zone name normalize_timestamp's fromisoformat can't
    # parse on its own; RecruiteeSource must convert it first.
    fx = fixture("recruitee")
    assert fx["offers"][0]["published_at"].endswith(" UTC")
    vacancies = await RecruiteeSource("bunq").fetch(FakeClient(fx))
    assert vacancies[0].posted_at is not None


async def test_teamtailor_description_needs_no_second_call():
    client = FakeClient(fixture("teamtailor"))
    vacancies = await TeamtailorSource("proxima").fetch(client)
    assert len(client.requests) == 1  # one GET, like Recruitee -- no detail call
    assert len(vacancies) == 2
    v = vacancies[0]
    assert v.source == "teamtailor" and v.company == "Proxima"
    assert "<" not in v.description and len(v.description) > 100
    assert v.location_city == "Chicago"
    assert v.location_country == "US"
    assert v.posted_at is not None
    assert v.url.startswith("https://proxima.teamtailor.com/jobs/")


async def test_teamtailor_joins_multiple_office_locations():
    # The live "Senior Azure Platform Engineer" posting lists four
    # jobLocation entries (Cardiff, London, Leeds, and Cardiff again with a
    # slightly different address) -- location_city/country take only the
    # first, but location_raw must not silently drop or duplicate the rest.
    fx = fixture("teamtailor")
    assert len(fx["items"][1]["_jobposting"]["jobLocation"]) == 4
    vacancies = await TeamtailorSource("proxima").fetch(FakeClient(fx))
    v = vacancies[1]
    assert v.location_city == "Cardiff" and v.location_country == "GB"
    assert v.location_raw == "Cardiff, London, Leeds"  # de-duplicated
    # No live tenant probed ever surfaced a remote signal -- None reflects a
    # real gap in the source data, not a missed field mapping.
    assert v.is_remote is None


def test_build_sources_wires_smartrecruiters_from_companies_yaml(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text("smartrecruiters:\n  - acme\n")
    config = Config(companies_path=path)
    sources = build_sources(config, only="smartrecruiters")
    assert len(sources) == 1
    assert isinstance(sources[0], SmartRecruitersSource)
    assert sources[0].company == "acme"


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


async def test_workingnomads_uses_url_as_source_id_and_is_always_remote():
    # No "id" field exists on a Working Nomads job (unlike RemoteOK/Remotive/
    # Jobicy), so the job's own url doubles as source_id, the same fallback
    # Himalayas uses for guid.
    client = FakeClient(fixture("workingnomads"))
    vacancies = await WorkingNomadsSource().fetch(client)
    assert vacancies
    v = vacancies[0]
    assert v.source == "workingnomads" and v.company and v.title
    assert v.source_id == v.url and v.url.startswith("https://www.workingnomads.com/")
    assert v.is_remote is True
    assert "<" not in v.description[:200]
    # pub_date arrives as ISO-8601 with an explicit offset, not UTC or epoch --
    # normalize_timestamp should convert it rather than dropping it.
    assert v.posted_at is not None
    assert len(client.requests) == 1


async def test_remotejobsorg_reads_nested_company_name_and_ignores_pagination():
    # pagination.total runs into the thousands; only the first (bounded) page
    # is ever fetched, so exactly one request happens and pagination is simply
    # not followed.
    client = FakeClient(fixture("remotejobsorg"))
    vacancies = await RemoteJobsOrgSource().fetch(client)
    assert vacancies
    v = vacancies[0]
    assert v.source == "remotejobsorg" and v.title and v.url
    fx = fixture("remotejobsorg")
    assert v.company == fx["data"][0]["company"]["name"]
    assert v.source_id == fx["data"][0]["id"]
    assert v.is_remote is True
    assert v.employment_type == "Full-time"
    assert "<" not in v.description[:200]
    assert len(client.requests) == 1


async def test_adzuna_scopes_location_country_to_the_queried_country():
    # A single country, single page: the fixture has 2 rows (< RESULTS_PER_PAGE
    # of 50), so pagination stops after one request.
    client = FakeClient(fixture("adzuna"))
    vacancies = await AdzunaSource("id", "key", countries=["gb"]).fetch(client)
    assert vacancies
    v = vacancies[0]
    assert v.source == "adzuna" and v.company and v.title and v.url
    # location.display_name ("Marlow, Buckinghamshire") has no country hint
    # normalize_location recognizes -- the queried country code is used instead.
    assert v.location_country == "GB"
    assert "<" not in v.description[:200]
    _, params = client.requests[0]
    assert params["app_id"] == "id" and params["app_key"] == "key" and params["what"] == "remote"
    assert len(client.requests) == 1


async def test_adzuna_queries_every_configured_country():
    client = FakeClient(fixture("adzuna"))
    vacancies = await AdzunaSource("id", "key", countries=["gb", "us", "de"]).fetch(client)
    assert len(client.requests) == 3
    assert {v.location_country for v in vacancies} == {"GB", "US", "DE"}


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


def test_build_sources_skips_adzuna_silently_when_unconfigured(tmp_path):
    config = Config(companies_path=tmp_path / "none.yaml")
    names = [s.name for s in build_sources(config)]
    assert "adzuna" not in names


def test_build_sources_requesting_adzuna_without_credentials_raises(tmp_path):
    config = Config(companies_path=tmp_path / "none.yaml")
    with pytest.raises(ValueError, match="adzuna"):
        build_sources(config, only="adzuna")


def test_build_sources_includes_adzuna_once_credentials_are_set(tmp_path):
    config = Config(
        companies_path=tmp_path / "none.yaml",
        adzuna_app_id="id",
        adzuna_app_key="key",
    )
    sources = build_sources(config)
    assert [s for s in sources if s.name == "adzuna"]


async def test_run_one_skips_the_closed_check_for_breadth_sources(store):
    """Arbeitnow/RemoteOK/Jobicy cover many companies per call -- there is no
    single company to diff against, so the check must not run for them."""
    payload = fixture("arbeitnow")
    payload["links"]["next"] = None
    sem, lock = asyncio.Semaphore(1), asyncio.Lock()

    result = await _run_one(ArbeitnowSource(), FakeClient(payload), store, sem, lock)

    assert result.closed == 0
