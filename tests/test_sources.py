"""Adapters run against payloads recorded from the live APIs."""

import asyncio
from datetime import datetime, timezone

import pytest
from conftest import FakeClient, fixture, fixture_text

from jobsearch.config import Config
from jobsearch.vacancies.dedupe import vacancy_id
from jobsearch.vacancies.ingest import _run_one, build_sources
from jobsearch.vacancies.sources.adzuna import AdzunaSource
from jobsearch.vacancies.sources.arbeitnow import ArbeitnowSource
from jobsearch.vacancies.sources.ats import (
    AshbySource,
    GreenhouseSource,
    LeverSource,
    PersonioSource,
    RecruiteeSource,
    SmartRecruitersSource,
    TeamtailorSource,
    WorkableSource,
)
from jobsearch.vacancies.sources.base import SourceError
from jobsearch.vacancies.sources.hackernews import HackerNewsSource
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
    assert vacancies[0].posted_at == datetime(2026, 9, 7, 12, 5, 26, tzinfo=timezone.utc)


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


async def test_personio_parses_xml_feed_and_joins_description_sections():
    client = FakeClient(fixture_text("personio.xml"))
    vacancies = await PersonioSource("getquin").fetch(client)
    assert len(client.requests) == 1  # descriptions ride inline, no detail call
    assert len(vacancies) == 2
    v = vacancies[0]
    assert v.source == "personio" and v.company == "getquin"
    assert v.source_id == "2597783"
    assert v.url == "https://getquin.jobs.personio.de/job/2597783"
    assert v.title == "Finanzbuchhalter (f/m/d) - Teilzeit"
    # Several <jobDescription> sections, each a heading plus CDATA HTML.
    assert "Was wir tun" in v.description and "getquin" in v.description
    assert "<" not in v.description and "CDATA" not in v.description
    assert v.location_raw == "Remote" and v.is_remote is True
    assert v.employment_type == "permanent"
    assert v.posted_at == datetime(2026, 4, 10, 12, 3, 36, tzinfo=timezone.utc)


async def test_personio_joins_additional_offices():
    vacancies = await PersonioSource("ottonova").fetch(
        FakeClient(fixture_text("personio_ottonova.xml"))
    )
    v = vacancies[0]
    assert v.location_raw == "München, Köln"
    assert v.location_city == "München" and v.location_country == "DE"
    # subcompany carries a legal-entity suffix ("... AG - 9698") that varies
    # per posting within one tenant; the tenant slug is the stable name.
    assert v.company == "ottonova"


async def test_personio_dead_tenant_redirect_fails_fast():
    # A tenant that doesn't exist 307-redirects to personio.com's marketing
    # site (which then 429s behind a bot checkpoint) -- the redirect itself is
    # the dead-slug signal, so it must fail without following or retrying.
    client = FakeClient("", status_code=307)
    with pytest.raises(SourceError):
        await PersonioSource("nosuchtenant").fetch(client)
    assert len(client.requests) == 1


def test_build_sources_wires_personio_from_companies_yaml(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text("personio:\n  - getquin\n")
    sources = build_sources(Config(companies_path=path), only="personio")
    assert len(sources) == 1 and isinstance(sources[0], PersonioSource)


def _hn_client(item=None):
    search, item = fixture("hn_search"), item or fixture("hn_item")
    return FakeClient(lambda n: search if n == 1 else item)


async def test_hackernews_reads_latest_who_is_hiring_thread():
    # The search also returns "Who wants to be hired?" (candidates, not jobs)
    # and last month's thread; only the newest hiring thread is read.
    client = _hn_client()
    vacancies = await HackerNewsSource().fetch(client)
    assert len(client.requests) == 2
    assert client.requests[1][0].endswith("/items/49522897")
    # 7 recorded comments, 2 of which have no "|"/"·" header and are skipped.
    assert len(vacancies) == 5
    assert all(v.source == "hackernews" and v.id for v in vacancies)


async def test_hackernews_parses_pipe_separated_header():
    v = (await HackerNewsSource().fetch(_hn_client()))[0]
    assert v.company == "Modash.io"
    assert v.title == "Senior Product Engineer"
    assert v.location_raw == "Remote (Europe)"
    assert v.is_remote is True
    assert v.employment_type == "Full-time"
    assert v.source_id == "49522903"
    assert v.url == "https://news.ycombinator.com/item?id=49522903"
    assert v.posted_at == datetime(2026, 9, 1, 15, 1, 54, tzinfo=timezone.utc)
    assert "Modash helps brands" in v.description and "<p>" not in v.description


async def test_hackernews_strips_urls_from_company_and_falls_back_on_title():
    by_id = {v.source_id: v for v in await HackerNewsSource().fetch(_hn_client())}
    smarkets = by_id["49523018"]           # "Smarkets (https://...) | Full Time | Hybrid ..."
    assert smarkets.company == "Smarkets"
    assert smarkets.title                  # no role segment, but never empty
    assert smarkets.is_remote is False     # "Hybrid - Onsite (London, UK)"
    assert by_id["49522989"].company == "Snout"
    assert by_id["49522989"].title == "Multiple Engineering + Product Roles"
    assert by_id["49523166"].title == "Senior Engineer (Platform)"
    # "REMOTE WILL BE IGNORED" mentions remote but means the opposite.
    assert by_id["49523830"].is_remote is False


async def test_hackernews_skips_deleted_comments():
    item = fixture("hn_item")
    item["children"][0]["text"] = None   # how Algolia serves a deleted comment
    vacancies = await HackerNewsSource().fetch(_hn_client(item))
    assert "49522903" not in {v.source_id for v in vacancies}


async def test_hackernews_without_hiring_thread_is_an_error():
    search = fixture("hn_search")
    search["hits"] = [h for h in search["hits"] if "wants to be hired" in h["title"]]
    with pytest.raises(SourceError):
        await HackerNewsSource().fetch(FakeClient(search))


def test_build_sources_includes_hackernews_as_breadth_source():
    sources = build_sources(Config(), only="hackernews")
    assert len(sources) == 1 and isinstance(sources[0], HackerNewsSource)
