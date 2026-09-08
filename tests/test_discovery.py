"""discover-ats: slug guessing, probing, and the companies.yaml writer."""

import yaml
from conftest import FakeMultiClient

from jobsearch.vacancies.discovery import (
    PROBEABLE_ATS,
    ProbeResult,
    discover,
    hits,
    probe,
    slug_variants,
    write_discoveries,
)


def _hit(platform, slug):
    return ProbeResult(platform, slug, ok=True, fetched=1)


def test_slug_variants_strips_legal_suffix_and_tries_separators():
    variants = slug_variants("Acme Corp")
    assert "acme" in variants
    assert "acme-corp" in variants  # suffix-kept fallback is still offered
    assert len(variants) == len(set(variants))  # no duplicates


def test_slug_variants_handles_punctuation_and_multi_word_suffix_strip():
    variants = slug_variants("Hello, World! GmbH")
    assert "hello-world" in variants
    assert "helloworld" in variants
    assert "hello-world-gmbh" in variants  # fallback keeps the legal suffix


def test_slug_variants_empty_name_yields_nothing():
    assert slug_variants("   ") == []


def test_slug_variants_respects_max_variants_cap():
    assert len(slug_variants("Hello World Company GmbH", max_variants=2)) == 2


def _greenhouse_hit(slug="acme"):
    return {
        f"boards-api.greenhouse.io/v1/boards/{slug}/jobs": (
            200,
            {"jobs": [{
                "id": 1, "title": "Engineer", "company_name": "Acme",
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                "location": {"name": "Berlin"}, "content": "<p>Do the work.</p>",
            }]},
        )
    }


async def test_probe_reports_a_hit_when_the_slug_is_live():
    client = FakeMultiClient(_greenhouse_hit())
    result = await probe(client, "greenhouse", "acme")
    assert result.hit is True and result.fetched == 1


async def test_probe_reports_no_hit_for_a_dead_slug():
    client = FakeMultiClient({})  # everything 404s
    result = await probe(client, "greenhouse", "nope", retries=1)
    assert result.hit is False and result.ok is False


async def test_discover_finds_the_one_live_combination_among_many_guesses():
    # "Acme" has no legal suffix, so slug_variants collapses to one candidate
    # ("acme") tried across every probeable platform -- exactly one is wired
    # to respond with a live posting.
    client = FakeMultiClient(_greenhouse_hit())
    results = await discover(["Acme"], client=client)
    assert len(results) == len(PROBEABLE_ATS)
    live = hits(results)
    assert [(r.platform, r.slug) for r in live] == [("greenhouse", "acme")]


async def test_discover_respects_the_platforms_filter():
    client = FakeMultiClient(_greenhouse_hit())
    results = await discover(["Acme"], platforms=["lever", "ashby"], client=client)
    assert {r.platform for r in results} == {"lever", "ashby"}
    assert hits(results) == []  # the only fixture hit is greenhouse, excluded here


COMPANIES_YAML = """\
greenhouse:
  - hellofresh
  - sumup

lever:
  - finn
"""


def test_write_discoveries_appends_new_slug_and_skips_existing(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(COMPANIES_YAML)

    found = [
        _hit("greenhouse", "hellofresh"),  # already present -- must be skipped
        _hit("greenhouse", "acme"),
        _hit("lever", "finn"),  # already present -- must be skipped
    ]
    written = write_discoveries(path, found, note="test batch")

    assert written == {"greenhouse": ["acme"]}
    reparsed = yaml.safe_load(path.read_text())
    assert set(reparsed["greenhouse"]) == {"hellofresh", "sumup", "acme"}
    assert reparsed["lever"] == ["finn"]
    assert "test batch" in path.read_text()


def test_write_discoveries_is_idempotent_on_rerun(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(COMPANIES_YAML)

    write_discoveries(path, [_hit("greenhouse", "acme")])
    second = write_discoveries(path, [_hit("greenhouse", "acme")])

    assert second == {}
    assert yaml.safe_load(path.read_text())["greenhouse"].count("acme") == 1


def test_write_discoveries_creates_a_new_platform_section(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(COMPANIES_YAML)

    written = write_discoveries(path, [_hit("smartrecruiters", "acme")])

    assert written == {"smartrecruiters": ["acme"]}
    reparsed = yaml.safe_load(path.read_text())
    assert reparsed["smartrecruiters"] == ["acme"]
    assert reparsed["greenhouse"] == ["hellofresh", "sumup"]  # untouched


def test_write_discoveries_no_hits_leaves_file_untouched(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(COMPANIES_YAML)
    before = path.read_text()

    written = write_discoveries(path, [_hit("greenhouse", "hellofresh")])

    assert written == {}
    assert path.read_text() == before
