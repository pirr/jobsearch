"""Every case here is a shape a live API actually returned."""

from conftest import fixture

from jobsearch.vacancies.normalize import normalize_location, normalize_timestamp, strip_html


def test_strips_double_escaped_html_from_arbeitnow():
    raw = fixture("arbeitnow")["data"][0]["description"]
    assert "&lt;" in raw  # the double-escaping this test exists for
    out = strip_html(raw)
    assert "&lt;" not in out and "<div" not in out
    assert len(out) > 500


def test_strips_plain_html_from_greenhouse():
    out = strip_html(fixture("greenhouse")["jobs"][0]["content"])
    assert "<p>" not in out and "&lt;" not in out
    assert len(out) > 500


def test_preserves_block_structure_as_newlines():
    lines = [l for l in strip_html("<p>One</p><p>Two</p>").splitlines() if l]
    assert lines == ["One", "Two"]
    assert "\n" in strip_html("<ul><li>a</li><li>b</li></ul>")


def test_empty_and_none():
    assert strip_html(None) == ""
    assert strip_html("") == ""


def test_location_country_token_is_never_the_city():
    # Greenhouse's "Remote, US, Georgia" must not yield city="US".
    raw, city, country, remote = normalize_location("Remote, US, Georgia")
    assert city is None
    assert country == "US"
    assert remote is True


def test_location_shapes():
    assert normalize_location("Berlin")[1:3] == ("Berlin", "DE")
    assert normalize_location({"name": "Munich, Germany"})[1:3] == ("Munich", "DE")
    assert normalize_location("Zurich, Switzerland")[2] == "CH"
    assert normalize_location(None) == ("", None, None, None)


def test_unknown_location_yields_none_not_a_guess():
    # A wrong country silently removes jobs the candidate could take, so a miss
    # must produce None rather than a plausible-looking guess.
    _, city, country, _ = normalize_location("Somewhereville")
    assert country is None
    assert city == "Somewhereville"


def test_timestamp_accepts_epoch_and_iso():
    epoch = normalize_timestamp(1788512113)
    iso = normalize_timestamp("2026-09-04T04:33:17-04:00")
    assert epoch.year == 2026
    assert iso.tzinfo is not None
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("not a date") is None
