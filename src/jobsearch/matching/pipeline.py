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


def rate_and_save(store: Store, profile: CandidateProfile, llm, row, profile_hash: str):
    """Rate one vacancy and persist the result -- the per-row step
    run_match's shortlist loop uses, factored out so a single vacancy can
    also be re-rated on demand (e.g. `jobsearch update-description`, after
    pasting in a fuller description than the source API provided) without
    re-running the whole shortlist and re-paying for every other vacancy on
    it."""
    outcome = rate_one(llm, profile, row)
    if not outcome.ok:
        store.save_rating(
            row["id"], profile_hash, getattr(llm, "name", "unknown"),
            failed=True, error=outcome.error,
        )
        return outcome

    penalty, notes = soft_penalty(profile, row)
    outcome.penalty = penalty
    outcome.penalty_notes = notes
    score = max(0, min(100, outcome.rating.score - penalty))
    # A hard blocker the model found is not a filter -- it is a heavy score
    # penalty, so the candidate still sees it and can disagree.
    if outcome.rating.hard_blockers:
        score = min(score, 39)
    outcome.final_score = score
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
    return outcome


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
    # preserving each bucket's relevance order. Reachable vacancies get most
    # of the shortlist; region-restricted, non-sponsoring ones only get a
    # small reserved floor of slots, not the bulk of it. A real
    # corpus has thousands of reachable vacancies -- comfortably more than
    # shortlist_size on its own -- so without a floor the restricted bucket
    # would never get a single slot, ever, ceasing to be "deprioritized" and
    # becoming permanently invisible instead.
    scores = retriever.rank(profile.searchable_text(), [row_text(r) for r in filtered.kept])
    by_relevance = [r for r, _ in sorted(
        zip(filtered.kept, scores), key=lambda pair: pair[1], reverse=True
    )]
    reachable, restricted = [], []
    for r in by_relevance:
        (reachable if eligibility.is_reachable(eligibility.classify(r)) else restricted).append(r)

    # A soft constraint filters.py can already check deterministically off
    # structured fields (remote/location/employment_type -- no LLM needed)
    # must not spend a shortlist slot ahead of a vacancy with none, even when
    # its raw retrieval relevance is higher: the LLM call that slot buys is
    # wasted on a vacancy whose score will just be docked for the same
    # violation afterward anyway. Stable sort preserves relevance order
    # within each group. Not a bucket with its own reserved floor like
    # restricted/reachable above -- unlike geo/visa restriction, there is no
    # "otherwise permanently invisible" risk here to protect against, so a
    # plain reorder is enough.
    reachable.sort(key=lambda r: soft_penalty(profile, r)[0] > 0)

    reserved = min(len(restricted), max(1, shortlist_size // 10)) if restricted else 0
    if reachable:
        # Never let the reserve eat the only slot(s) reachable candidates
        # would otherwise get -- the floor exists so restricted vacancies
        # aren't permanently invisible, not to invert the priority order.
        reserved = min(reserved, max(0, shortlist_size - 1))
    reachable_take = min(len(reachable), shortlist_size - reserved)
    leftover = shortlist_size - reachable_take - reserved
    restricted_take = min(len(restricted), reserved + max(0, leftover))
    shortlist = reachable[:reachable_take] + restricted[:restricted_take]

    # `--force` promises to re-rate every vacancy already rated for this
    # profile. Honor that even for one the bucketing above would otherwise
    # leave out (e.g. a vacancy rated before this eligibility logic existed,
    # now reclassified as restricted) instead of silently limiting a forced
    # re-rate to whatever the freshly computed shortlist happens to contain.
    if force:
        by_id = {r["id"]: r for r in filtered.kept}
        shortlisted_ids = {r["id"] for r in shortlist}
        shortlist += [
            by_id[vid] for vid in store.rated_ids(profile_hash)
            if vid in by_id and vid not in shortlisted_ids
        ]

    report.shortlisted = len(shortlist)
    report.restricted_total = len(restricted)
    report.restricted_shortlisted = sum(
        1 for r in shortlist if not eligibility.is_reachable(eligibility.classify(r))
    )

    done = set() if force else store.rated_ids(profile_hash)
    for row in shortlist:
        if row["id"] in done:
            report.already_rated += 1
            continue

        outcome = rate_and_save(store, profile, llm, row, profile_hash)
        report.llm_calls += 1
        if not outcome.ok:
            report.failed += 1
        else:
            report.rated += 1
            report.dropped_claims += len(outcome.dropped_claims)
        if progress:
            progress(row, outcome)
    return report
