"""Guards the default_factory fix: config must re-read the environment on
every load_config() call, not freeze values at import time."""

import importlib


def test_load_config_rereads_environment(monkeypatch):
    from jobsearch import config as config_mod
    importlib.reload(config_mod)

    monkeypatch.setenv("JOBSEARCH_THRESHOLD", "77")
    monkeypatch.setenv("JOBSEARCH_LLM", "fake")
    assert config_mod.load_config().score_threshold == 77
    assert config_mod.load_config().llm_backend == "fake"

    monkeypatch.setenv("JOBSEARCH_THRESHOLD", "10")
    assert config_mod.load_config().score_threshold == 10


def test_bad_int_env_reports_the_variable_name(monkeypatch):
    from jobsearch import config as config_mod

    monkeypatch.setenv("JOBSEARCH_SHORTLIST", "not-a-number")
    try:
        config_mod.load_config()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "JOBSEARCH_SHORTLIST" in str(exc)
