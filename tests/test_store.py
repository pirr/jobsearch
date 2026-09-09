import json

import pytest
from conftest import make_vacancy


def test_upsert_is_idempotent(store):
    v = make_vacancy()
    assert store.upsert_vacancies([v]) == (1, 0)
    assert store.upsert_vacancies([v]) == (0, 1)
    assert store.count_vacancies() == 1


def test_get_vacancy_returns_the_row(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    row = store.get_vacancy(v.id)
    assert row["id"] == v.id and row["title"] == v.title


def test_get_vacancy_returns_none_when_missing(store):
    assert store.get_vacancy("does-not-exist") is None


def test_update_description_replaces_the_stored_text(store):
    # For manually pasting the full posting text after clicking through to
    # it by hand (e.g. an Adzuna listing, whose API only ever returns a
    # ~500-char snippet) -- one vacancy at a time, never automated.
    v = make_vacancy(description="short snippet")
    store.upsert_vacancies([v])
    store.update_description(v.id, "full real description with much more detail")
    assert store.get_vacancy(v.id)["description"] == "full real description with much more detail"


def test_ranked_includes_evidence_with_penalty_notes(store):
    # evidence (including soft_penalty's penalty_notes) is written but was
    # never selected back by ranked() -- `jobsearch show --full` silently had
    # nothing to render even though the data existed in the DB.
    v = make_vacancy()
    store.upsert_vacancies([v])
    evidence = {
        "summary_fit": "strong Python match",
        "penalty_notes": ["-10 location Manchester != Remote EU [inferred, not stated in CV]"],
    }
    store.save_rating(v.id, "h", "fake", score=70, verdict="good", evidence=evidence)

    row = store.ranked("h", 0, 10)[0]
    assert json.loads(row["evidence"])["penalty_notes"] == evidence["penalty_notes"]


def test_mark_duplicate_hides_it_from_ranked_but_not_the_canonical_one(store):
    # dedupe.py's hash-based identity misses this in practice -- e.g. a
    # typo'd company name ("doctorlib" vs "Doctolib") or a differently
    # formatted location_raw ("Berlin" vs "Berlin, Berlin, Germany") produces
    # two different ids for what is really one posting. This is the manual
    # escape hatch for that.
    dup = make_vacancy(source_id="1", company="doctorlib", location_raw="Berlin")
    canonical = make_vacancy(source_id="2", company="Doctolib",
                              location_raw="Berlin, Berlin, Germany")
    store.upsert_vacancies([dup, canonical])
    store.save_rating(dup.id, "h", "f", score=90)
    store.save_rating(canonical.id, "h", "f", score=85)

    store.mark_duplicate(dup.id, canonical.id)

    ids = {r["id"] for r in store.ranked("h", 0, 10)}
    assert dup.id not in ids
    assert canonical.id in ids


def test_mark_duplicate_transfers_status_to_a_canonical_with_none_yet(store):
    # A real bug found in production: a vacancy already marked "applied" was
    # then marked a duplicate. The canonical vacancy took over its visibility
    # but not its status -- the fact that the candidate had applied silently
    # disappeared from `show` along with the hidden duplicate.
    dup = make_vacancy(source_id="1")
    canonical = make_vacancy(source_id="2", title="Different Title")
    store.upsert_vacancies([dup, canonical])
    store.set_status(dup.id, "applied")

    store.mark_duplicate(dup.id, canonical.id)

    store.save_rating(canonical.id, "h", "f", score=80)
    assert store.ranked("h", 0, 10)[0]["status"] == "applied"


def test_mark_duplicate_does_not_silently_overwrite_a_conflicting_status(store):
    dup = make_vacancy(source_id="1")
    canonical = make_vacancy(source_id="2", title="Different Title")
    store.upsert_vacancies([dup, canonical])
    store.set_status(dup.id, "applied")
    store.set_status(canonical.id, "dismissed")

    note = store.mark_duplicate(dup.id, canonical.id)

    assert note is not None and "applied" in note and "dismissed" in note
    store.save_rating(canonical.id, "h", "f", score=80)
    assert store.ranked("h", 0, 10, only_status="dismissed")[0]["status"] == "dismissed"


def test_unmark_duplicate_restores_visibility(store):
    dup = make_vacancy(source_id="1")
    canonical = make_vacancy(source_id="2", title="Different Title")
    store.upsert_vacancies([dup, canonical])
    store.save_rating(dup.id, "h", "f", score=90)
    store.mark_duplicate(dup.id, canonical.id)

    store.unmark_duplicate(dup.id)

    assert dup.id in {r["id"] for r in store.ranked("h", 0, 10)}


def test_mark_duplicate_rejects_marking_a_vacancy_as_its_own_duplicate(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    with pytest.raises(ValueError, match="itself"):
        store.mark_duplicate(v.id, v.id)


def test_ratings_are_keyed_by_profile_hash(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, "hash-a", "fake", score=90, verdict="strong")
    assert store.rated_ids("hash-a") == {v.id}
    # An edited profile has a different hash, so it re-rates rather than
    # reusing a score computed for a different candidate.
    assert store.rated_ids("hash-b") == set()


def test_failed_ratings_are_recorded_not_dropped(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, "h", "fake", failed=True, error="bad json")
    assert store.failed_ratings("h") == 1
    assert store.rated_ids("h") == set()  # a failure must not count as done


def test_dismissed_vacancies_leave_the_pipeline(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, "h", "fake", score=95, verdict="strong")
    assert len(store.ranked("h", 60, 10)) == 1
    store.set_status(v.id, "dismissed")
    assert store.ranked("h", 60, 10) == []
    assert store.all_vacancies(exclude_dismissed=True) == []


def test_closed_vacancies_are_excluded_from_ranked_by_default(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, "h", "fake", score=95, verdict="strong")
    assert len(store.ranked("h", 60, 10)) == 1
    store.set_status(v.id, "closed")
    assert store.ranked("h", 60, 10) == []
    assert store.ranked("h", 60, 10, only_status="closed")[0]["id"] == v.id


def test_close_missing_vacancies_marks_gone_ones_closed(store):
    v1 = make_vacancy(source_id="1", title="Role One")
    v2 = make_vacancy(source_id="2", title="Role Two")
    store.upsert_vacancies([v1, v2])
    store.save_rating(v1.id, "h", "f", score=80)
    store.save_rating(v2.id, "h", "f", score=80)

    closed = store.close_missing_vacancies("test", "Acme", present_source_ids={"1"})

    assert closed == 1
    assert [r["title"] for r in store.ranked("h", 0, 10)] == ["Role One"]
    assert store.ranked("h", 0, 10, only_status="closed")[0]["title"] == "Role Two"


def test_close_missing_vacancies_never_overwrites_applied(store):
    """Applying is worth knowing even after a role closes -- a later
    disappearance from the source must not silently erase it."""
    v = make_vacancy(source_id="1")
    store.upsert_vacancies([v])
    store.set_status(v.id, "applied")

    closed = store.close_missing_vacancies("test", "Acme", present_source_ids=set())

    assert closed == 0
    store.save_rating(v.id, "h", "f", score=80)
    assert store.ranked("h", 0, 10)[0]["status"] == "applied"


def test_close_missing_vacancies_is_idempotent(store):
    v = make_vacancy(source_id="1")
    store.upsert_vacancies([v])
    first = store.close_missing_vacancies("test", "Acme", present_source_ids=set())
    second = store.close_missing_vacancies("test", "Acme", present_source_ids=set())
    assert first == 1
    assert second == 0  # already closed, not re-counted


def test_close_missing_vacancies_scoped_to_source_and_company(store):
    other_company = make_vacancy(source_id="1", company="OtherCo", title="Other Co Role")
    target = make_vacancy(source_id="1", company="Acme", title="Acme Role")
    store.upsert_vacancies([other_company, target])

    closed = store.close_missing_vacancies("test", "Acme", present_source_ids=set())

    assert closed == 1
    store.save_rating(other_company.id, "h", "f", score=80)
    store.save_rating(target.id, "h", "f", score=80)
    assert store.ranked("h", 0, 10, only_status="closed")[0]["title"] == "Acme Role"


def test_ranked_respects_threshold_and_order(store):
    lo, hi = make_vacancy(title="Low", description="a"), make_vacancy(title="High", description="b")
    store.upsert_vacancies([lo, hi])
    store.save_rating(lo.id, "h", "f", score=45)
    store.save_rating(hi.id, "h", "f", score=88)
    rows = store.ranked("h", 60, 10)
    assert [r["title"] for r in rows] == ["High"]


def test_ingest_runs_record_failure(store):
    run = store.start_run("greenhouse:dead")
    store.finish_run(run, ok=False, count=0, error="404 Not Found")
    latest = store.recent_runs(1)[0]
    assert latest["ok"] == 0 and "404" in latest["error"]


def test_profile_records_list_with_rated_count(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_profile_record("20260904-100000-cv", "hash-a", ["cv.pdf"])
    store.save_rating(v.id, "hash-a", "fake", score=90, verdict="strong")

    rows = store.list_profiles()
    assert len(rows) == 1
    assert rows[0]["id"] == "20260904-100000-cv"
    assert rows[0]["rated_count"] == 1


def test_profile_with_no_ratings_has_zero_count(store):
    store.save_profile_record("20260904-100000-cv", "hash-a", ["cv.pdf"])
    assert store.list_profiles()[0]["rated_count"] == 0


def test_failed_ratings_do_not_count_as_rated_for_a_profile(store):
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_profile_record("p1", "hash-a", ["cv.pdf"])
    store.save_rating(v.id, "hash-a", "fake", failed=True, error="bad json")
    assert store.list_profiles()[0]["rated_count"] == 0


def test_two_archives_of_the_same_profile_hash_both_get_the_full_count(store):
    # Re-archiving an unchanged CV is expected to produce two rows sharing
    # one profile_hash, each showing the same rated_count for that hash.
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_profile_record("p1", "hash-a", ["cv.pdf"])
    store.save_profile_record("p2", "hash-a", ["cv.pdf"])
    store.save_rating(v.id, "hash-a", "fake", score=90, verdict="strong")
    counts = {r["id"]: r["rated_count"] for r in store.list_profiles()}
    assert counts == {"p1": 1, "p2": 1}


def test_get_profile_returns_none_for_unknown_id(store):
    assert store.get_profile("nope") is None


def test_get_profile_returns_the_record(store):
    store.save_profile_record("id-1", "hash-a", ["cv.pdf", "cover.pdf"])
    row = store.get_profile("id-1")
    assert row["profile_hash"] == "hash-a"
    assert json.loads(row["source_files"]) == ["cv.pdf", "cover.pdf"]


# ---- ranked_page (keyset pagination) --------------------------------------

def test_ranked_page_keyset_is_stable_when_a_row_leaves_the_view_mid_browse(store, profile):
    # The classic OFFSET-pagination bug: if v2 is marked dismissed between
    # fetching page 1 and page 2, an OFFSET-based query would shift v3 into
    # page 1's old slot and duplicate it into page 2. Keyset pagination
    # anchors each page on the last row actually seen, not a row count, so
    # this can't happen.
    h = profile.hash()
    v1 = make_vacancy(source_id="1", title="A")
    v2 = make_vacancy(source_id="2", title="B")
    v3 = make_vacancy(source_id="3", title="C")
    store.upsert_vacancies([v1, v2, v3])
    store.save_rating(v1.id, h, "f", score=90)
    store.save_rating(v2.id, h, "f", score=80)
    store.save_rating(v3.id, h, "f", score=70)

    page1 = store.ranked_page(h, 0, page_size=2)
    assert [r["id"] for r in page1] == [v1.id, v2.id]

    store.set_status(v2.id, "dismissed")

    cursor = (page1[-1]["_sort_key"], page1[-1]["id"])
    page2 = store.ranked_page(h, 0, after=cursor, page_size=2)
    assert [r["id"] for r in page2] == [v3.id]


def test_ranked_page_defaults_to_active_view_score_desc(store, profile):
    h = profile.hash()
    lo, hi = make_vacancy(title="Low"), make_vacancy(title="High", source_id="2")
    store.upsert_vacancies([lo, hi])
    store.save_rating(lo.id, h, "f", score=45)
    store.save_rating(hi.id, h, "f", score=88)
    store.set_status(lo.id, "dismissed")

    rows = store.ranked_page(h, 0)
    assert [r["id"] for r in rows] == [hi.id]


def test_ranked_page_only_status_shows_just_that_bucket(store, profile):
    h = profile.hash()
    v = make_vacancy()
    store.upsert_vacancies([v])
    store.save_rating(v.id, h, "f", score=90)
    store.set_status(v.id, "dismissed")

    assert store.ranked_page(h, 0) == []
    assert [r["id"] for r in store.ranked_page(h, 0, only_status="dismissed")] == [v.id]


def test_ranked_page_sorts_by_requested_column(store, profile):
    h = profile.hash()
    a = make_vacancy(source_id="1", company="Zeta", title="A")
    b = make_vacancy(source_id="2", company="Alpha", title="B")
    store.upsert_vacancies([a, b])
    store.save_rating(a.id, h, "f", score=50)
    store.save_rating(b.id, h, "f", score=90)

    by_company_asc = store.ranked_page(h, 0, sort_by="company", sort_dir="asc")
    assert [r["id"] for r in by_company_asc] == [b.id, a.id]  # Alpha before Zeta

    by_title_desc = store.ranked_page(h, 0, sort_by="title", sort_dir="desc")
    assert [r["id"] for r in by_title_desc] == [b.id, a.id]  # "B" before "A"


def test_ranked_page_puts_unposted_dates_last_regardless_of_direction(store, profile):
    from datetime import datetime, timezone

    h = profile.hash()
    dated = make_vacancy(source_id="1", title="Dated",
                          posted_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    undated = make_vacancy(source_id="2", title="Undated")
    store.upsert_vacancies([dated, undated])
    store.save_rating(dated.id, h, "f", score=50)
    store.save_rating(undated.id, h, "f", score=50)

    desc = store.ranked_page(h, 0, sort_by="posted_at", sort_dir="desc")
    asc = store.ranked_page(h, 0, sort_by="posted_at", sort_dir="asc")
    assert [r["id"] for r in desc][-1] == undated.id
    assert [r["id"] for r in asc][-1] == undated.id


def test_ranked_page_keyset_pagination_works_for_non_default_sort(store, profile):
    h = profile.hash()
    names = ["Charlie", "Alpha", "Bravo"]
    ids = []
    for i, name in enumerate(names):
        v = make_vacancy(source_id=str(i), company=name)
        store.upsert_vacancies([v])
        store.save_rating(v.id, h, "f", score=50)
        ids.append(v.id)

    page1 = store.ranked_page(h, 0, sort_by="company", sort_dir="asc", page_size=2)
    assert [r["company"] for r in page1] == ["Alpha", "Bravo"]
    cursor = (page1[-1]["_sort_key"], page1[-1]["id"])
    page2 = store.ranked_page(h, 0, sort_by="company", sort_dir="asc", after=cursor, page_size=2)
    assert [r["company"] for r in page2] == ["Charlie"]


def test_ranked_page_rejects_unknown_sort_by(store, profile):
    with pytest.raises(ValueError, match="sort_by"):
        store.ranked_page(profile.hash(), 0, sort_by="bogus")


def test_ranked_page_rejects_unknown_sort_dir(store, profile):
    with pytest.raises(ValueError, match="sort_dir"):
        store.ranked_page(profile.hash(), 0, sort_dir="sideways")
