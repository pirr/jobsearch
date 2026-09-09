import sqlite3

import pytest
from conftest import make_vacancy

from jobsearch.candidate.profile import CandidateProfile, Constraint
from jobsearch.matching import eligibility
from jobsearch.matching.evaluation import ndcg_at_k, precision_at_k, recall_at_k
from jobsearch.matching.filters import apply_hard_filters, soft_penalty
from jobsearch.matching.llm.base import parse_json_response, LLMError
from jobsearch.matching.llm.fake import FakeLLM
from jobsearch.matching.pipeline import run_match
from jobsearch.matching.rating import Rating, rate_one, verify_grounding
from jobsearch.matching.retrieval import BM25Retriever


def row(**kw):
    base = dict(id="v1", title="Backend Engineer", company="Acme", description="Python work",
                location_raw="Berlin", location_city="Berlin", location_country="DE",
                is_remote=None, employment_type=None)
    base.update(kw)
    return base


# ---- filters ---------------------------------------------------------

def test_stated_constraint_excludes():
    p = CandidateProfile(constraints=[Constraint(
        kind="remote", value="remote only", stated=True, evidence="remote only please")])
    result = apply_hard_filters(p, [row(is_remote=0), row(id="v2", is_remote=1)])
    assert [r["id"] for r in result.kept] == ["v2"]
    assert "not remote" in result.excluded[0][1]


def test_inferred_constraint_never_excludes():
    """The whole point of the hard/soft split."""
    p = CandidateProfile(constraints=[Constraint(kind="remote", value="remote only", stated=False)])
    result = apply_hard_filters(p, [row(is_remote=0)])
    assert len(result.kept) == 1
    assert result.excluded == []
    penalty, notes = soft_penalty(p, row(is_remote=0))
    assert penalty > 0 and "inferred" in notes[0]


def test_unknown_remote_status_is_not_treated_as_no():
    # is_remote is None when the source never said. Unknown must not exclude.
    p = CandidateProfile(constraints=[Constraint(
        kind="remote", value="remote only", stated=True, evidence="remote only please")])
    assert len(apply_hard_filters(p, [row(is_remote=None)]).kept) == 1


def test_remote_vacancy_satisfies_a_location_constraint():
    p = CandidateProfile(constraints=[Constraint(
        kind="location", value="Berlin", stated=True, evidence="based in Berlin")])
    assert len(apply_hard_filters(p, [row(location_raw="Lisbon", is_remote=1)]).kept) == 1


def test_no_hard_constraints_keeps_everything():
    assert len(apply_hard_filters(CandidateProfile(), [row(), row(id="v2")]).kept) == 2


def test_unknown_remote_status_does_not_trigger_the_location_penalty():
    """Two ATS boards can report is_remote differently for the literal same
    posting (one says True, the other never says at all) -- unknown must not
    be treated as confirmed on-site for scoring any more than it is for hard
    filtering, or the same job scores inconsistently depending only on which
    board it was fetched from."""
    p = CandidateProfile(constraints=[Constraint(
        kind="location", value="Budva, Montenegro", stated=False)])
    penalty, notes = soft_penalty(p, row(location_raw="Berlin", is_remote=None))
    assert penalty == 0
    assert notes == []


def test_confirmed_onsite_still_triggers_the_location_penalty():
    p = CandidateProfile(constraints=[Constraint(
        kind="location", value="Budva, Montenegro", stated=False)])
    penalty, notes = soft_penalty(p, row(location_raw="Berlin", is_remote=False))
    assert penalty == 10
    assert "inferred" in notes[0]


# ---- geo/visa eligibility --------------------------------------------

def test_classify_flags_us_restricted_remote_with_denied_sponsorship():
    r = row(location_raw="Remote", is_remote=True,
            description="This salary range takes into account all possible "
                        "locations within the United States. Visa sponsorship "
                        "is not offered for this role.")
    elig = eligibility.classify(r)
    assert elig.region_restricted == "US"
    assert elig.visa_sponsorship is False
    assert not eligibility.is_reachable(elig)


def test_classify_treats_jobicy_europe_geo_as_eu_restricted():
    r = row(location_raw="Europe", is_remote=True, description="Full-time remote role.")
    elig = eligibility.classify(r)
    assert elig.region_restricted == "EU"
    assert not eligibility.is_reachable(elig)


def test_classify_treats_jobicy_anywhere_geo_as_open():
    r = row(location_raw="Anywhere", is_remote=True, description="Contract role.")
    elig = eligibility.classify(r)
    assert elig.region_restricted is None
    assert eligibility.is_reachable(elig)


def test_classify_region_restricted_but_sponsoring_is_reachable():
    r = row(location_raw="Remote (USA)", is_remote=True,
            description="We are happy to sponsor a work visa for the right candidate.")
    elig = eligibility.classify(r)
    assert elig.region_restricted == "US"
    assert elig.visa_sponsorship is True
    assert eligibility.is_reachable(elig)


def test_classify_ignores_restriction_phrasing_on_a_declared_onsite_role():
    """is_remote=False routes through the existing soft location penalty
    instead -- this module only judges the scope of an actually-remote role."""
    r = row(location_raw="Berlin", is_remote=False,
            description="Remote (USA) team members occasionally visit this office.")
    elig = eligibility.classify(r)
    assert elig.region_restricted is None
    assert eligibility.is_reachable(elig)


# ---- retrieval -------------------------------------------------------

def test_bm25_ranks_the_relevant_document_first():
    docs = ["Marketing manager for campaigns",
            "Senior Python backend engineer, Kubernetes and PostgreSQL",
            "Payroll specialist"]
    top = BM25Retriever().top_n("Python backend Kubernetes", docs, 1)
    assert top[0][0] == 1


def test_bm25_handles_empty_corpus():
    assert BM25Retriever().rank("python", []) == []


# ---- rating grounding ------------------------------------------------

def test_ungrounded_claims_are_dropped():
    r = Rating(score=90, matched_skills=[
        {"skill": "Python", "evidence_cv": "Senior Python engineer with Django",
         "evidence_vacancy": "you will write Python services"},
        {"skill": "Rust", "evidence_cv": "expert in Rust", "evidence_vacancy": "Rust required"},
    ])
    dropped = verify_grounding(r, "Senior Python engineer with Django in Berlin",
                               "Here you will write Python services daily")
    assert [m.skill for m in r.matched_skills] == ["Python"]
    assert dropped == ["Rust"]


def test_grounding_tolerates_reflowed_whitespace_and_smart_quotes():
    r = Rating(score=50, matched_skills=[
        {"skill": "Python", "evidence_cv": "Senior   Python\n engineer",
         "evidence_vacancy": "we don't mind"}])
    assert verify_grounding(r, "Senior Python engineer", "we don’t mind") == []


def test_score_bounds_are_enforced():
    with pytest.raises(Exception):
        Rating.model_validate({"score": 150})


def test_json_salvaged_from_fenced_and_prose_responses():
    assert parse_json_response('```json\n{"score": 7}\n```') == {"score": 7}
    assert parse_json_response('Sure: {"score": 7} done') == {"score": 7}
    with pytest.raises(LLMError):
        parse_json_response("no json here at all")


def test_json_salvaged_when_unrelated_braces_precede_the_real_answer():
    """Reproduces a real failure: a model with no separate reasoning field
    writes its chain of thought into the same text as the final answer. If
    that reasoning mentions a brace character first, a naive
    find("{")...rfind("}") grabs from the wrong opening brace onward and
    fails to parse at all -- this is exactly what happened once against a
    real local model, and an identical retry produced a different, clean
    response, which is why the fix cannot rely on that non-determinism."""
    text = (
        'The role wants a candidate with a "{skills}" style match, something '
        'like {"not": "the answer"} maybe, but let me think it through.\n'
        'Final answer:\n'
        '{"score": 82, "matched_skills": [{"skill": "Python"}]}'
    )
    assert parse_json_response(text) == {
        "score": 82, "matched_skills": [{"skill": "Python"}]
    }


def test_json_salvaged_when_a_quoted_value_contains_braces():
    # A job posting could plausibly quote a template literal like ${var} --
    # a brace inside a JSON string must never be mistaken for nesting.
    # Wrapped in prose so the direct json.loads(text) attempt fails first and
    # this actually exercises the string-aware salvage scan, not just the
    # fast path for already-valid JSON.
    text = (
        'Thinking about it: {this is not json}\n'
        '{"score": 50, "summary_fit": "uses template literals like ${var} in examples"}'
    )
    assert parse_json_response(text) == {
        "score": 50, "summary_fit": "uses template literals like ${var} in examples"
    }


def test_json_salvage_prefers_the_larger_candidate_over_position():
    # Not "last wins" -- size is the real signal. Position alone can't
    # distinguish a stray fragment mentioned before the answer from one
    # trailing after it (see the two tests above and below); size can,
    # because our schemas are always richer than a stray fragment.
    text = '{"score": 1, "extra": "bigger"} some text {"score": 2}'
    assert parse_json_response(text) == {"score": 1, "extra": "bigger"}


def test_json_salvage_equal_size_ties_prefer_the_later_span():
    text = '{"score": 1} some text in between {"score": 2}'
    assert parse_json_response(text) == {"score": 2}


def test_json_salvaged_when_a_fragment_trails_after_the_real_answer():
    """Reproduces a second real failure, distinct from the first: the model
    generated the full, correct answer, then appended a stray duplicate of
    one matched_skills entry afterward (a known degenerate-generation
    pattern). A naive "last complete object wins" rule -- the fix for the
    FIRST failure -- picks that trailing fragment instead of the real
    answer, which is exactly what happened: parsing succeeded but produced
    a bare {"skill": ...} dict instead of a Rating, failing schema
    validation with `score: Field required`. Size, not position, is what
    correctly prefers the real answer in both directions."""
    text = (
        '{"score": 88, "matched_skills": [{"skill": "Docker"}], '
        '"missing_skills": [], "summary_fit": "fits well", "summary_gap": ""}'
        '\n{"skill": "Docker", "evidence_vacancy": "duplicated fragment"}'
    )
    result = parse_json_response(text)
    assert result["score"] == 88
    assert "skill" not in result


# ---- evaluation metrics ---------------------------------------------

def test_ndcg_distinguishes_order_where_precision_cannot():
    good = [True] * 5 + [False] * 5
    bad = [False] * 5 + [True] * 5
    assert precision_at_k(good, 10) == precision_at_k(bad, 10)
    assert ndcg_at_k(good, 10) > ndcg_at_k(bad, 10)


def test_recall_uses_the_full_relevant_count():
    assert recall_at_k([True, False], 2, total_relevant=4) == 0.25


# ---- pipeline end to end --------------------------------------------

def test_pipeline_calls_the_llm_only_for_the_shortlist(store, profile):
    """The cost guarantee: LLM calls are bounded by shortlist size, not corpus size."""
    vacancies = [make_vacancy(title=f"Engineer {i}", description=f"Python role number {i}",
                              source_id=str(i)) for i in range(30)]
    store.upsert_vacancies(vacancies)
    llm = FakeLLM([{"score": 70, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""} for _ in range(100)])

    report = run_match(store, profile, llm, BM25Retriever(), shortlist_size=5)

    assert report.total == 30
    assert report.shortlisted == 5
    assert report.llm_calls == 5
    assert len(llm.calls) == 5


def test_rerunning_match_does_not_re_pay(store, profile):
    store.upsert_vacancies([make_vacancy(source_id=str(i), title=f"Role {i}") for i in range(3)])
    responses = [{"score": 80, "matched_skills": []} for _ in range(20)]

    first = run_match(store, profile, FakeLLM(list(responses)), BM25Retriever(), shortlist_size=3)
    second = run_match(store, profile, FakeLLM(list(responses)), BM25Retriever(), shortlist_size=3)

    assert first.llm_calls == 3
    assert second.llm_calls == 0
    assert second.already_rated == 3


def test_force_re_rates(store, profile):
    store.upsert_vacancies([make_vacancy()])
    run_match(store, profile, FakeLLM([{"score": 80}]), BM25Retriever(), shortlist_size=1)
    again = run_match(store, profile, FakeLLM([{"score": 80}]), BM25Retriever(),
                      shortlist_size=1, force=True)
    assert again.llm_calls == 1


def test_hard_blockers_cap_the_score(store, profile):
    store.upsert_vacancies([make_vacancy()])
    llm = FakeLLM([{"score": 95, "hard_blockers": ["requires a medical licence"],
                    "matched_skills": []}])
    run_match(store, profile, llm, BM25Retriever(), shortlist_size=1)
    assert store.ranked(profile.hash(), 0, 10)[0]["score"] <= 39


def test_malformed_llm_output_is_recorded_as_failed(store, profile):
    store.upsert_vacancies([make_vacancy()])
    llm = FakeLLM([{"score": "not a number"}, {"score": "still bad"}])
    report = run_match(store, profile, llm, BM25Retriever(), shortlist_size=1)
    assert report.failed == 1
    assert store.failed_ratings(profile.hash()) == 1
    assert store.ranked(profile.hash(), 0, 10) == []


def test_restricted_vacancies_are_shortlisted_only_after_reachable_ones(store, profile):
    restricted = make_vacancy(
        source_id="us-only", title="Backend Engineer",
        description="Python role. Remote (USA). Visa sponsorship is not offered for this role.",
        is_remote=True,
    )
    reachable = make_vacancy(
        source_id="open", title="Backend Engineer",
        description="Python role, remote from anywhere.",
        is_remote=True,
    )
    # Inserted restricted-first so a naive relevance-only ranking could still
    # land it first; the reachable one must win the single shortlist slot.
    store.upsert_vacancies([restricted, reachable])
    llm = FakeLLM([{"score": 70, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""} for _ in range(5)])

    report = run_match(store, profile, llm, BM25Retriever(), shortlist_size=1)

    assert report.restricted_total == 1
    assert report.restricted_shortlisted == 0
    assert report.rated == 1
    assert store.ranked(profile.hash(), 0, 10)[0]["id"] == reachable.id


def test_restricted_vacancies_backfill_leftover_shortlist_slots(store, profile):
    restricted = make_vacancy(
        source_id="us-only", title="Backend Engineer",
        description="Python role. Remote (USA). Visa sponsorship is not offered for this role.",
        is_remote=True,
    )
    store.upsert_vacancies([restricted])
    llm = FakeLLM([{"score": 70, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""}])

    report = run_match(store, profile, llm, BM25Retriever(), shortlist_size=3)

    assert report.restricted_total == 1
    assert report.restricted_shortlisted == 1
    assert report.rated == 1


def test_restricted_vacancies_still_get_a_reserved_floor_with_a_large_reachable_pool(store, profile):
    """Without a reserved floor, any real corpus -- where the reachable pool
    alone dwarfs shortlist_size -- would starve the restricted bucket
    completely, turning "deprioritized" into "never rated." This is the bug
    this test guards against."""
    restricted = make_vacancy(
        source_id="us-only", title="Backend Engineer",
        description="Python role. Remote (USA). Visa sponsorship is not offered for this role.",
        is_remote=True,
    )
    reachable_pool = [make_vacancy(source_id=f"open-{i}", title=f"Backend Engineer {i}",
                                    description=f"Python role, remote from anywhere, number {i}.",
                                    is_remote=True)
                       for i in range(50)]
    store.upsert_vacancies([restricted, *reachable_pool])

    report = run_match(store, profile,
                       FakeLLM([{"score": 70, "matched_skills": []} for _ in range(20)]),
                       BM25Retriever(), shortlist_size=10)

    assert report.restricted_total == 1
    assert report.restricted_shortlisted == 1
    assert restricted.id in store.rated_ids(profile.hash())


def test_soft_location_violation_does_not_win_a_shortlist_slot_over_a_satisfying_vacancy(store):
    """Mirrors test_restricted_vacancies_are_shortlisted_only_after_reachable_ones,
    but for filters.py's soft constraints (checkable deterministically off
    structured fields, no LLM needed) instead of eligibility.py's geo/visa
    ones: a vacancy violating a soft location preference must not spend the
    one shortlist slot ahead of a satisfying vacancy, even when its raw
    retrieval relevance is higher (verified directly: BM25 scores the
    wrong-location posting 3.84 vs. 0.77 for the right-location one, purely
    from denser keyword overlap)."""
    p = CandidateProfile(
        headline="Senior Python Backend Engineer",
        skills=["Python", "Django", "Kubernetes", "PostgreSQL"],
        constraints=[Constraint(kind="location", value="Berlin", stated=False)],
    )
    wrong_location = make_vacancy(
        source_id="nyc", title="Senior Python Backend Engineer", location_raw="New York",
        description="Python Django Kubernetes PostgreSQL backend engineer. "
                     "Python Django Kubernetes PostgreSQL Python Django.",
        is_remote=False,
    )
    right_location = make_vacancy(
        source_id="berlin", title="Backend Engineer", location_raw="Berlin",
        description="Python backend role.",
        is_remote=False,
    )
    # Inserted wrong-location-first so a naive relevance-only ranking could
    # still land it first; the satisfying one must win the single slot.
    store.upsert_vacancies([wrong_location, right_location])
    llm = FakeLLM([{"score": 70, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""} for _ in range(5)])

    report = run_match(store, p, llm, BM25Retriever(), shortlist_size=1)

    assert report.rated == 1
    assert store.ranked(p.hash(), 0, 10)[0]["id"] == right_location.id


def test_force_reconsiders_a_restricted_vacancy_crowded_out_by_reachable_ones(store, profile):
    """Reproduces the real bug: a vacancy classified restricted, and already
    rated before this eligibility logic existed, must still be reachable via
    --force even though normal bucketing gives the restricted bucket 0
    shortlist slots whenever a single reachable vacancy already fills a
    shortlist this small."""
    restricted = make_vacancy(
        source_id="us-only", title="Backend Engineer",
        description="Python role. Remote (USA). Visa sponsorship is not offered for this role.",
        is_remote=True,
    )
    reachable = make_vacancy(
        source_id="open", title="Backend Engineer",
        description="Python role, remote from anywhere.",
        is_remote=True,
    )
    store.upsert_vacancies([restricted, reachable])
    store.save_rating(restricted.id, profile.hash(), "test", score=92, verdict="strong")
    store.save_rating(reachable.id, profile.hash(), "test", score=70, verdict="worth applying")

    # Normal run: the single slot goes to the reachable vacancy; both are
    # already rated, so nothing gets re-rated.
    normal = run_match(store, profile, FakeLLM([]), BM25Retriever(), shortlist_size=1)
    assert normal.llm_calls == 0

    # --force must still reconsider the restricted vacancy despite it never
    # making the shortlist.
    forced = run_match(
        store, profile,
        FakeLLM([{"score": 25, "hard_blockers": ["US-only remote"], "matched_skills": []},
                 {"score": 65, "matched_skills": []}]),
        BM25Retriever(), shortlist_size=1, force=True,
    )
    assert forced.llm_calls == 2
    row = next(r for r in store.ranked(profile.hash(), 0, 10) if r["id"] == restricted.id)
    assert row["score"] != 92  # actually re-rated, not left at the seeded value


def test_rate_one_prompt_includes_candidate_constraints_and_geo_signal(profile):
    llm = FakeLLM([{"score": 50, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""}])
    r = row(location_raw="Remote", is_remote=True,
            description="Remote (USA). Visa sponsorship is not offered for this role.")

    rate_one(llm, profile, r)

    _, prompt = llm.calls[0]
    assert "Berlin" in prompt  # the candidate's own location constraint
    assert "restricted to US" in prompt
    assert "explicitly denied" in prompt


def test_rate_one_prompt_discourages_hard_blocking_plain_onsite_mismatch(profile):
    """A plain on-site/hybrid posting (no geo/visa restriction detected) must
    not read as license to hard-block on location alone -- that regressed
    into treating the candidate's inferred location as an exclusion, which
    this project's hard/soft constraint split explicitly says an inferred
    constraint must never do."""
    llm = FakeLLM([{"score": 50, "matched_skills": [], "missing_skills": [],
                    "summary_fit": "", "summary_gap": ""}])
    r = row(location_raw="Berlin", is_remote=None,
            description="Join our Berlin office. Hybrid role, 3 days on-site.")

    rate_one(llm, profile, r)

    _, prompt = llm.calls[0]
    assert "ordinary on-site/hybrid role" in prompt
    assert "not treat this vacancy's plain location as a hard blocker" in prompt


def test_soft_penalty_lowers_but_does_not_remove(store):
    p = CandidateProfile(skills=["Python"],
                         constraints=[Constraint(kind="remote", value="remote only", stated=False)])
    store.upsert_vacancies([make_vacancy(is_remote=False)])
    run_match(store, p, FakeLLM([{"score": 90, "matched_skills": []}]), BM25Retriever(),
              shortlist_size=1)
    rows = store.ranked(p.hash(), 0, 10)
    assert len(rows) == 1
    assert rows[0]["score"] == 78  # 90 - 12


# ---- embedding retriever resilience ----------------------------------

class _ContextLimitedLLM:
    """Stands in for a server whose embedding model has a small context
    window: any single input over `limit` chars raises, matching the real
    "exceed_context_size_error" failure this test reproduces."""

    name = "context-limited"

    def __init__(self, limit=200):
        self.limit = limit
        self.batch_calls = []

    def embed(self, texts, *, role="document"):
        self.batch_calls.append(len(texts))
        if any(len(t) > self.limit for t in texts):
            from jobsearch.matching.llm.base import LLMError
            raise LLMError("exceeds context size")
        return [[float(len(t))] for t in texts]


class _RoleRecordingLLM:
    """Records which role each embed() call was made with, so rank()'s query
    vs. document handling (including the per-item fallback path) can be
    checked without a real embedding model."""

    name = "role-recording"

    def __init__(self):
        self.roles: list[tuple[str, str]] = []  # (text, role)

    def embed(self, texts, *, role="document"):
        self.roles.extend((t, role) for t in texts)
        return [[float(len(t))] for t in texts]


def test_embedding_retriever_embeds_query_and_documents_with_the_right_role():
    from jobsearch.matching.retrieval import EmbeddingRetriever

    llm = _RoleRecordingLLM()
    retriever = EmbeddingRetriever(llm, batch_size=10)

    retriever.rank("my cv", ["doc one", "doc two"])

    assert ("my cv", "query") in llm.roles
    assert ("doc one", "document") in llm.roles
    assert ("doc two", "document") in llm.roles
    assert not any(role == "query" for text, role in llm.roles if text != "my cv")


class _RoleRecordingContextLimitedLLM(_RoleRecordingLLM):
    """Combines _ContextLimitedLLM's failure behavior with role recording, to
    confirm the shrink-retry fallback path also embeds with role="document"."""

    def __init__(self, limit=200):
        super().__init__()
        self.limit = limit

    def embed(self, texts, *, role="document"):
        vectors = super().embed(texts, role=role)
        if any(len(t) > self.limit for t in texts):
            from jobsearch.matching.llm.base import LLMError
            raise LLMError("exceeds context size")
        return vectors


def test_embedding_retriever_fallback_path_also_uses_document_role():
    from jobsearch.matching.retrieval import EmbeddingRetriever

    llm = _RoleRecordingContextLimitedLLM(limit=200)
    retriever = EmbeddingRetriever(llm, batch_size=10, max_chars=1500)
    docs = ["short doc"] * 3 + ["x" * 1000]  # forces the batch call to fail

    retriever.rank("query", docs)

    doc_roles = {role for text, role in llm.roles if text != "query"}
    assert doc_roles == {"document"}
    assert ("query", "query") in llm.roles


def test_embedding_retriever_falls_back_per_item_on_batch_context_error():
    from jobsearch.matching.retrieval import EmbeddingRetriever

    llm = _ContextLimitedLLM(limit=200)
    retriever = EmbeddingRetriever(llm, batch_size=10, max_chars=1500)
    docs = ["short doc"] * 5 + ["x" * 1000]  # one doc exceeds the server's limit

    scores = retriever.rank("query", docs)

    assert len(scores) == 6
    # The five short documents still get real scores despite the one bad
    # document poisoning the batch call.
    assert all(s > 0 for s in scores[:5])


def test_embedding_retriever_degrades_to_zero_not_a_crash_for_unembeddable_text():
    from jobsearch.matching.retrieval import EmbeddingRetriever

    llm = _ContextLimitedLLM(limit=30)  # even a shrunk doc will not fit
    retriever = EmbeddingRetriever(llm, batch_size=10, max_chars=1500)
    scores = retriever.rank("q", ["y" * 2000])

    assert scores == [0.0]  # degrades silently rather than raising
