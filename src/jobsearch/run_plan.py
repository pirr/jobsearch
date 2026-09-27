"""The plan `jobsearch run` follows: ingest once, then match several profiles.

It lives in a YAML file (data/pipeline.yaml, or JOBSEARCH_PIPELINE) rather
than in environment variables like the rest of the config, because each
profile can carry its own overrides. Profiles are referenced by their
archive id (`jobsearch profiles`), so a run never touches the active
profile.json that `show`/`tui` read.

Validation is strict on purpose: `run` is the command left to churn through a
long ingest and many LLM calls unattended, so a typo like `shortlst:` must
fail up front, not be silently ignored an hour in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PLAN_KEYS = {"ingest", "profiles"}
_PROFILE_KEYS = {"id", "shortlist", "force"}


class RunPlanError(ValueError):
    """The plan file is missing or malformed."""


@dataclass(frozen=True)
class ProfileRun:
    id: str
    shortlist: int | None = None   # None -> JOBSEARCH_SHORTLIST
    force: bool = False


@dataclass(frozen=True)
class RunPlan:
    ingest: bool = True
    profiles: tuple[ProfileRun, ...] = ()


def load_run_plan(path: Path) -> RunPlan:
    path = Path(path)
    if not path.exists():
        raise RunPlanError(
            f"{path} not found -- copy data/pipeline.example.yaml there and list "
            "profile ids from `jobsearch profiles`"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RunPlanError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RunPlanError(f"{path}: expected a mapping with a `profiles:` list")

    unknown = set(data) - _PLAN_KEYS
    if unknown:
        raise RunPlanError(f"{path}: unknown key(s) {', '.join(sorted(unknown))}")

    ingest = data.get("ingest", True)
    if not isinstance(ingest, bool):
        raise RunPlanError(f"{path}: `ingest` must be true or false, got {ingest!r}")

    entries = data.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise RunPlanError(f"{path}: `profiles` must be a non-empty list of profile ids")

    profiles = tuple(_profile(path, entry) for entry in entries)
    seen: set[str] = set()
    for p in profiles:
        if p.id in seen:
            raise RunPlanError(f"{path}: duplicate profile id {p.id!r}")
        seen.add(p.id)
    return RunPlan(ingest=ingest, profiles=profiles)


def _profile(path: Path, entry) -> ProfileRun:
    # A bare string is shorthand for {id: ...} -- the common case needs no
    # overrides.
    if isinstance(entry, str):
        entry = {"id": entry}
    if not isinstance(entry, dict):
        raise RunPlanError(f"{path}: each profile must be an id or a mapping, got {entry!r}")

    unknown = set(entry) - _PROFILE_KEYS
    if unknown:
        raise RunPlanError(f"{path}: unknown profile key(s) {', '.join(sorted(unknown))}")

    profile_id = entry.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise RunPlanError(f"{path}: every profile needs an `id` (see `jobsearch profiles`)")

    shortlist = entry.get("shortlist")
    # bool is an int subclass; `shortlist: true` is a typo, not 1.
    if shortlist is not None and (
        not isinstance(shortlist, int) or isinstance(shortlist, bool) or shortlist < 1
    ):
        raise RunPlanError(f"{path}: {profile_id}: `shortlist` must be a positive integer")

    force = entry.get("force", False)
    if not isinstance(force, bool):
        raise RunPlanError(f"{path}: {profile_id}: `force` must be true or false")

    return ProfileRun(id=profile_id.strip(), shortlist=shortlist, force=force)
