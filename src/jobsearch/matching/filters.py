"""Hard filters and soft penalties.

Only constraints the CV stated outright (with a verified quote) exclude a
vacancy. Inferred constraints become score penalties instead, because an
inferred constraint that is wrong would otherwise delete good matches with no
trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..candidate.profile import CandidateProfile, Constraint

# Soft penalties are deliberately modest: they reorder a shortlist, they do not
# bury an otherwise excellent match on the strength of a guess.
SOFT_PENALTY = {"remote": 12, "location": 10, "employment_type": 6, "seniority": 8}


@dataclass
class FilterResult:
    kept: list
    excluded: list[tuple[str, str]]  # (vacancy_id, reason)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)


def _wants_remote(value: str) -> bool:
    v = value.lower()
    return "remote" in v and "not" not in v and "no " not in v


def _violates(constraint: Constraint, row) -> str | None:
    """Reason this vacancy breaks the constraint, or None."""
    kind, value = constraint.kind, constraint.value
    if kind == "remote" and _wants_remote(value):
        # Only exclude on a definite negative. `None` means the source never
        # said, and unknown is not the same as "no".
        if row["is_remote"] == 0:
            return f"not remote (candidate stated: {value})"
        return None
    if kind == "location":
        wanted = value.strip().lower()
        if not wanted:
            return None
        if row["is_remote"] == 1:
            return None  # remote satisfies any location requirement
        haystack = " ".join(
            str(row[f] or "").lower()
            for f in ("location_raw", "location_city", "location_country")
        )
        if wanted and wanted not in haystack:
            return f"location {row['location_raw'] or 'unknown'} != {value}"
        return None
    if kind == "employment_type":
        et = (row["employment_type"] or "").lower()
        if et and value.strip().lower() not in et:
            return f"employment type {row['employment_type']} != {value}"
        return None
    # seniority, language and visa are not reliably present in source data, so
    # they are left to the LLM rating stage rather than guessed at here.
    return None


def apply_hard_filters(profile: CandidateProfile, rows) -> FilterResult:
    hard = profile.hard_constraints
    if not hard:
        return FilterResult(list(rows), [])
    kept, excluded = [], []
    for row in rows:
        reason = next((r for c in hard if (r := _violates(c, row))), None)
        if reason:
            excluded.append((row["id"], reason))
        else:
            kept.append(row)
    return FilterResult(kept, excluded)


def soft_penalty(profile: CandidateProfile, row) -> tuple[int, list[str]]:
    """Points to subtract, with reasons, for unmet inferred preferences."""
    total, notes = 0, []
    for constraint in profile.soft_constraints:
        reason = _violates(constraint, row)
        if reason:
            points = SOFT_PENALTY.get(constraint.kind, 0)
            if points:
                total += points
                notes.append(f"-{points} {reason} [inferred, not stated in CV]")
    return total, notes
