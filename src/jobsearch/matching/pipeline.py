"""The match stage: filter -> shortlist -> rate -> store.

Reads vacancies the ingest stage wrote and writes ratings the show stage reads.
The stages never call each other; SQLite is the only thing between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..candidate.profile import CandidateProfile
from ..vacancies.store import Store
from . import eligibility
from .filters import apply_hard_filters, soft_penalty
from .rating import rate_one, verdict_for
from .retrieval import Retriever


def row_text(row) -> str:
    parts = [row["title"], row["company"], row["location_raw"], row["description"]]
    return "\n".join(p for p in parts if p)


@dataclass
class MatchReport:
    total: int = 0
    after_filters: int = 0
    excluded: list[tuple[str, str]] = field(default_factory=list)
    shortlisted: int = 0
    restricted_total: int = 0
    restricted_shortlisted: int = 0
    already_rated: int = 0
    rated: int = 0
    failed: int = 0
    dropped_claims: int = 0
    llm_calls: int = 0


def run_match(
    store: Store,
    profile: CandidateProfile,
    llm,
    retriever: Retriever,
    *,
    shortlist_size: int,
    force: bool = False,
    progress=None,
) -> MatchReport:
    report = MatchReport()
    profile_hash = profile.hash()

    rows = store.all_vacancies(exclude_dismissed=True)
    report.total = len(rows)
    if not rows:
        return report

    filtered = apply_hard_filters(profile, rows)
    report.after_filters = len(filtered.kept)
    report.excluded = filtered.excluded
    if not filtered.kept:
        return report

    # Shortlist cheaply. This is the step that keeps a run affordable: without
    # it the LLM would be called once per vacancy across the whole corpus.
    #
    # Rank once over the whole filtered corpus (one consistent set of BM25/
    # embedding statistics), then bucket by geo/visa reachability while
    # preserving each bucket's relevance order. Reachable vacancies fill the
    # shortlist first; region-restricted, non-sponsoring ones only backfill
    # whatever slots are left, instead of competing for LLM budget on equal
    # footing with a vacancy the candidate can actually take.
    scores = retriever.rank(profile.searchable_text(), [row_text(r) for r in filtered.kept])
    by_relevance = [r for r, _ in sorted(
        zip(filtered.kept, scores), key=lambda pair: pair[1], reverse=True
    )]
    reachable, restricted = [], []
    for r in by_relevance:
        (reachable if eligibility.is_reachable(eligibility.classify(r)) else restricted).append(r)
    shortlist = (reachable + restricted)[:shortlist_size]
    report.shortlisted = len(shortlist)
    report.restricted_total = len(restricted)
    report.restricted_shortlisted = max(0, len(shortlist) - len(reachable))

    done = set() if force else store.rated_ids(profile_hash)
    for row in shortlist:
        if row["id"] in done:
            report.already_rated += 1
            continue

        outcome = rate_one(llm, profile, row)
        report.llm_calls += 1

        if not outcome.ok:
            report.failed += 1
            store.save_rating(
                row["id"], profile_hash, getattr(llm, "name", "unknown"),
                failed=True, error=outcome.error,
            )
        else:
            penalty, notes = soft_penalty(profile, row)
            score = max(0, min(100, outcome.rating.score - penalty))
            # A hard blocker the model found is not a filter -- it is a heavy
            # score penalty, so the candidate still sees it and can disagree.
            if outcome.rating.hard_blockers:
                score = min(score, 39)
            report.rated += 1
            report.dropped_claims += len(outcome.dropped_claims)
            store.save_rating(
                row["id"], profile_hash, getattr(llm, "name", "unknown"),
                score=score,
                verdict=verdict_for(score),
                matched_skills=[m.model_dump() for m in outcome.rating.matched_skills],
                missing_skills=outcome.rating.missing_skills,
                hard_blockers=outcome.rating.hard_blockers,
                evidence={
                    "summary_fit": outcome.rating.summary_fit,
                    "summary_gap": outcome.rating.summary_gap,
                    "soft_penalty": penalty,
                    "penalty_notes": notes,
                    "dropped_claims": outcome.dropped_claims,
                },
            )
        if progress:
            progress(row, outcome)
    return report
