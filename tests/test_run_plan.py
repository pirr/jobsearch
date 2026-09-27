"""The `jobsearch run` plan file: parsing and validation."""

import pytest

from jobsearch.run_plan import ProfileRun, RunPlan, RunPlanError, load_run_plan


def _write(tmp_path, text):
    path = tmp_path / "pipeline.yaml"
    path.write_text(text)
    return path


def test_loads_profiles_with_overrides(tmp_path):
    plan = load_run_plan(_write(tmp_path, """
ingest: false
profiles:
  - id: 20260101-000000-backend_cv
  - id: 20260102-000000-platform_cv
    shortlist: 100
    force: true
"""))
    assert plan == RunPlan(
        ingest=False,
        profiles=(
            ProfileRun(id="20260101-000000-backend_cv"),
            ProfileRun(id="20260102-000000-platform_cv", shortlist=100, force=True),
        ),
    )


def test_ingest_defaults_on_and_bare_string_entries_are_ids(tmp_path):
    plan = load_run_plan(_write(tmp_path, "profiles:\n  - some-id\n"))
    assert plan.ingest is True
    assert plan.profiles == (ProfileRun(id="some-id"),)


def test_missing_file_points_at_the_example(tmp_path):
    with pytest.raises(RunPlanError, match="pipeline.example.yaml"):
        load_run_plan(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text, message", [
    ("", "profiles"),
    ("profiles: []\n", "profiles"),
    ("profiles: some-id\n", "profiles"),
    ("ingest: yes-please\nprofiles: [a]\n", "ingest"),
    ("profiles: [a]\nshortlist: 5\n", "shortlist"),                # top-level typo
    ("profiles:\n  - id: a\n    shortlst: 5\n", "shortlst"),        # entry typo
    ("profiles:\n  - shortlist: 5\n", "id"),
    ("profiles:\n  - id: a\n    shortlist: 0\n", "shortlist"),
    ("profiles:\n  - id: a\n    shortlist: many\n", "shortlist"),
    ("profiles:\n  - id: a\n    force: maybe\n", "force"),
    ("profiles: [a, a]\n", "duplicate"),
])
def test_rejects_malformed_plans(tmp_path, text, message):
    # A typo must fail loudly before a long ingest, not be silently ignored.
    with pytest.raises(RunPlanError, match=message):
        load_run_plan(_write(tmp_path, text))
