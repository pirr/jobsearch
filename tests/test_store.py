import json

from conftest import make_vacancy


def test_upsert_is_idempotent(store):
    v = make_vacancy()
    assert store.upsert_vacancies([v]) == (1, 0)
    assert store.upsert_vacancies([v]) == (0, 1)
    assert store.count_vacancies() == 1


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
