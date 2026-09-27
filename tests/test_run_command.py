"""`jobsearch run`: ingest once, then match every profile in the plan file."""

import pytest
from conftest import make_vacancy
from typer.testing import CliRunner

from jobsearch.cli.main import app
from jobsearch.vacancies.store import Store


@pytest.fixture
def env(tmp_path, monkeypatch, profile):
    """A DB with a few vacancies and two archived profiles, fake LLM, bm25."""
    storage = tmp_path / "profile_storage"
    other = profile.model_copy(update={"headline": "Senior Data Engineer"})
    ids = {"p-backend": profile, "p-data": other}
    for profile_id, candidate in ids.items():
        (storage / profile_id).mkdir(parents=True)
        candidate.save(storage / profile_id / "profile.json")

    db = tmp_path / "test.db"
    with Store(db) as store:
        store.upsert_vacancies([
            make_vacancy(source_id=str(i), title=f"Python Engineer {i}", is_remote=True)
            for i in range(3)
        ])

    monkeypatch.setenv("JOBSEARCH_DB", str(db))
    monkeypatch.setenv("JOBSEARCH_PROFILE", str(tmp_path / "profile.json"))
    monkeypatch.setenv("JOBSEARCH_PROFILE_STORAGE", str(storage))
    monkeypatch.setenv("JOBSEARCH_PIPELINE", str(tmp_path / "pipeline.yaml"))
    monkeypatch.setenv("JOBSEARCH_LLM", "fake")
    monkeypatch.setenv("JOBSEARCH_RETRIEVER", "bm25")
    return tmp_path, db, ids


def test_run_rates_every_listed_profile_without_touching_the_active_one(env):
    tmp_path, db, ids = env
    (tmp_path / "pipeline.yaml").write_text(
        "ingest: false\nprofiles:\n  - p-backend\n  - id: p-data\n    shortlist: 2\n"
    )
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    with Store(db) as store:
        assert len(store.rated_ids(ids["p-backend"].hash())) == 3
        assert len(store.rated_ids(ids["p-data"].hash())) == 2   # per-profile shortlist
        assert store.recent_runs() == []                          # ingest: false
    assert not (tmp_path / "profile.json").exists()
    assert "run summary" in result.output


def test_run_rejects_unknown_profile_before_ingesting(env):
    tmp_path, db, _ = env
    (tmp_path / "pipeline.yaml").write_text("profiles:\n  - p-backend\n  - no-such-id\n")
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 1
    assert "no-such-id" in result.output
    with Store(db) as store:
        assert store.recent_runs() == []   # failed fast: no source was fetched
        assert store.rated_ids(env[2]["p-backend"].hash()) == set()


def test_run_skip_ingest_flag_overrides_the_file(env, monkeypatch):
    tmp_path, db, ids = env
    (tmp_path / "pipeline.yaml").write_text("ingest: true\nprofiles: [p-backend]\n")

    def boom(*args, **kwargs):
        raise AssertionError("ingest must not run with --skip-ingest")

    monkeypatch.setattr("jobsearch.cli.main._ingest_and_report", boom)
    result = CliRunner().invoke(app, ["run", "--skip-ingest"])
    assert result.exit_code == 0, result.output
    with Store(db) as store:
        assert len(store.rated_ids(ids["p-backend"].hash())) == 3


def test_run_ingests_when_the_file_says_so(env, monkeypatch):
    tmp_path, _, _ = env
    (tmp_path / "pipeline.yaml").write_text("profiles: [p-backend]\n")
    calls = []
    monkeypatch.setattr(
        "jobsearch.cli.main._ingest_and_report", lambda config, sources: calls.append(sources)
    )
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0]   # real source list, network stubbed out
