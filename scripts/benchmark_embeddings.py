#!/usr/bin/env python3
"""Compare embedding models across several retrieval scenarios at once.

`jobsearch eval` already answers "how good is retrieval quality" for one
configured embedding model, one golden set at a time -- comparing two models
means re-running it by hand with different env vars, which is what a manual
nomic-embed-text-v1.5 vs. embeddinggemma-300m comparison turned into. This
script does that fan-out directly: several models x several scenarios, one
table, in one run.

It is a manual, network-dependent dev tool, not part of the pytest suite --
it needs a real embedding server reachable at --host and is not offline like
every test in tests/. It reuses jobsearch's real production code
(LlamaCppClient, EmbeddingRetriever, BM25Retriever, and the exact scoring
functions matching/evaluation.py uses for `jobsearch eval`) rather than
reimplementing any of it, so results here are guaranteed to reflect what the
app itself would actually do, not an approximation of it.

Two kinds of scenario:
  - Golden sets (data/golden/*.json) -- real candidate profiles against real,
    hand-labelled vacancies. This is where the reliable signal is. Across 5
    real profiles (backend, devops/sre, data-engineer, frontend, general
    python-developer), nomic-embed-text-v1.5 never loses on p@k/recall@k and
    wins clearly on the two most infra/systems-heavy profiles (backend,
    devops); embeddinggemma-300m ties on two others and is marginally ahead
    on one nDCG figure (frontend). Add more golden sets for broader
    real-world coverage; `GoldenSet.load_dir` already picks up every file
    there.
  - Synthetic cases (below) -- small, fast, hand-built probes. They don't
    replace golden-set evidence (see the limitation noted below), but one of
    them earned its place: inspecting raw cosine similarities for
    "negation-robustness" showed nomic-embed-text-v1.5 ranking a "no
    Kubernetes experience necessary" posting *above* a genuine Kubernetes
    match, 0.7714 vs 0.6757 in the ML/NLP version that first surfaced this --
    a real, decisive miss caused by literal token overlap with the query
    ("ML", "Kubernetes") overriding an explicit negation, not noise from a
    too-small sample. "jargon-synonyms" and "seniority-vs-domain" remain too
    easy at their current size -- every model, even bm25, aces them, so they
    currently confirm baseline competence rather than differentiate models.
    Treat the synthetic cases as a fast smoke test overall, not a substitute
    for golden-set evidence.

Usage:
    python scripts/benchmark_embeddings.py
    python scripts/benchmark_embeddings.py --host http://localhost:8080 --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jobsearch.matching.evaluation import (  # noqa: E402
    GoldenSet,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from jobsearch.matching.llm.llamacpp import LlamaCppClient  # noqa: E402
from jobsearch.matching.retrieval import BM25Retriever, EmbeddingRetriever  # noqa: E402


@dataclass
class ModelConfig:
    label: str
    embed_model: str
    query_prefix: str = ""
    document_prefix: str = ""


# Add a candidate model here to include it in the comparison. `embed_model`
# is the id llama-swap (or a bare llama-server's --alias) routes on; prefixes
# come from that model's own card -- see .env.example for why they matter.
DEFAULT_MODELS = [
    ModelConfig("nomic-embed-text-v1.5", "nomic-embed-text-v1.5",
                "search_query: ", "search_document: "),
    ModelConfig("embeddinggemma-300m", "embeddinggemma-300m",
                "task: search result | query: ", "title: none | text: "),
]


@dataclass
class Case:
    """One labelled retrieval scenario: a query against several documents,
    each marked relevant or not -- same shape as a golden set, just inline
    instead of a file, for scenarios cheap enough not to need curated data."""
    name: str
    query: str
    documents: list[tuple[str, bool]]  # (text, is_relevant)


def synthetic_cases() -> list[Case]:
    return [
        Case(
            name="jargon-synonyms",
            query="Senior Python Backend Engineer, Kubernetes, distributed systems",
            documents=[
                ("Backend Engineer needed, strong Python, must know k8s and "
                 "container orchestration", True),
                ("Senior Software Engineer, Python, running services on "
                 "Kubernetes clusters", True),
                ("Frontend Developer, React and TypeScript, no backend "
                 "experience needed", False),
                ("Pastry Chef, ten years experience in patisserie and baking", False),
            ],
        ),
        Case(
            name="seniority-vs-domain",
            # Modeled on the pattern that broke embeddinggemma-300m on the
            # real golden set (a same-seniority, wrong-domain posting
            # outranking true matches) -- but at this size/cleanliness both
            # models currently solve it fine, so it isn't reproducing that
            # failure yet. Left in as a baseline-competence check; needs
            # noisier/closer distractors to actually differentiate.
            query="Senior Python Backend Engineer, Django, FastAPI, PostgreSQL",
            documents=[
                ("Senior Backend Engineer, Python, Django, building REST APIs", True),
                ("Senior Product Manager, investment and pension products, "
                 "fintech", False),
                ("Senior TypeScript Developer, Node.js, frontend and backend", False),
                ("Principal Security Engineer, cybersecurity architecture, "
                 "cloud infrastructure", False),
                ("Staff Software Engineer, Python, PostgreSQL, high-scale APIs", True),
            ],
        ),
        Case(
            name="abbreviation-expansion",
            # The "no ML experience required" negative used to live here.
            # Investigating why nomic-embed-text-v1.5 failed this case
            # (checked raw cosine similarities directly) found it wasn't an
            # abbreviation-expansion failure at all: nomic ranked that
            # document *above* the true match, 0.7714 vs 0.6757, because it
            # shares the literal token "ML" with the query despite the
            # negation. That's a different, more specific failure mode --
            # split out below as its own case -- so this one no longer
            # contains that confound and now cleanly tests only whether
            # "ML"/"NLP" get matched to their spelled-out forms.
            query="ML Engineer with NLP experience",
            documents=[
                ("Machine Learning Engineer, natural language processing, "
                 "transformers", True),
                ("Data Engineer, SQL and Spark, ETL pipelines", False),
                ("Sales Manager, enterprise software, B2B", False),
            ],
        ),
        Case(
            name="negation-robustness",
            # Isolates the failure mode found above: does a "no X required"
            # / "X not needed" document get correctly ranked LOW despite
            # literally containing the query's key term? A model doing
            # lexical-overlap-flavored matching (rather than encoding the
            # negation) will rank it too high. Uses a different skill/domain
            # (Kubernetes/AWS, not ML/NLP) than the case above so this isn't
            # just one example that happened to trip on it.
            #
            # Both models currently pass this on p@k/recall@k/nDCG -- but
            # checking raw cosine similarities shows why that's not the same
            # as "fixed": the negation-trap document scores 0.7435 (nomic) /
            # 0.7254 (embeddinggemma), against 0.7746/0.7784 for the weaker
            # real match and 0.5970/0.6004 for a genuinely unrelated
            # document. That's a ~0.03-0.05 margin, not a comfortable one --
            # close enough that a slightly different phrasing could flip the
            # ranking. The ML/NLP case above DID flip, with only one true
            # match to protect it. Read a "pass" here as "not currently
            # broken," not "negation is handled well."
            query="Senior Backend Engineer with Kubernetes and AWS experience",
            documents=[
                ("Backend Engineer role, hands-on production Kubernetes and "
                 "AWS required", True),
                ("Senior Platform Engineer, running Kubernetes clusters on "
                 "AWS at scale", True),
                ("Backend Engineer, no Kubernetes experience necessary, we "
                 "will train you on the job", False),
                ("Frontend Developer, React and TypeScript, uses AWS S3 for "
                 "static hosting only", False),
            ],
        ),
    ]


def realistic_cases() -> list[Case]:
    """Same idea as synthetic_cases(), but built from real vacancy text (full
    ~2000-char postings from state/jobsearch.db) instead of hand-written
    sentences -- the synthetic cases' own limitation, documented above, is
    that short clean text doesn't carry the noise real postings do. Stored
    separately in scripts/realistic_cases.json (NOT under data/golden/ --
    GoldenSet.load_dir() globs every *.json there, so a case file dropped in
    that directory breaks every golden-set-driven run with a KeyError;
    verified this the hard way before moving it here).

    Cases so far:

    "negation-robustness-real" / "negation-robustness-real-frontend" reuse
    the synthetic "negation-robustness" idea (a hedged/optional skill mention
    that shouldn't count as a real match) with real postings found by
    searching the corpus for "nice to have"/"not required" phrasing near a
    skill -- e.g. a Flix "Automation Solutions Engineer (Celonis)" role whose
    actual description says "Scripting (Python or similar) is a plus but not
    required". This confirmed a dilution hypothesis directly: both models
    pass at real ~2000-char length, with a noticeably wider safety margin
    than the short synthetic version (weakest-match-to-trap gap 0.031/0.053
    nomic/embeddinggemma on the short Kubernetes/AWS case vs. 0.079/0.088 at
    real length -- roughly 2-2.5x wider). One buried hedge sentence among
    ~2000 characters of otherwise clearly-different content pulls the whole
    document's embedding further from the query than the same sentence does
    when it's most of a short document. Short synthetic cases aren't just
    "less noisy" than real postings -- they can specifically understate how
    safe a model actually is against this failure mode.

    "language-robustness" -- found by ranking every golden set's true
    positives by their own retrieval rank and inspecting the worst ones: a
    genuinely relevant, entirely-German posting ("Softwarearchitekt") ranked
    dead last (32/32) under nomic-embed-text-v1.5 in the
    senior-python-backend-berlin golden set, purely from the language
    mismatch. embeddinggemma-300m does notably better on the same document
    (19/32) but neither is reliable in a tight, controlled version of this
    case.

    "remote-preference-real" -- CandidateProfile.searchable_text() (what
    actually reaches the embedding query in production, see
    candidate/profile.py) never includes the candidate's remote/location
    constraints at all -- filters.py's soft "remote" penalty (12 points) is
    the only place remote preference is enforced, and it only runs AFTER an
    LLM rating call has already been spent on the shortlisted vacancy. This
    case checks whether that gap actually costs anything at the retrieval
    step: two real remote Python postings vs. two real on-site-only ones
    (one, GlassFlow Berlin, has strong enough content overlap to be a
    genuine temptation), query text WITH an explicit remote-preference
    sentence added (the *best* case for the model, better than what
    production actually sends). nomic-embed-text-v1.5 still ranks both
    remote postings above both on-site ones correctly. embeddinggemma-300m
    does not: both on-site postings outrank one of the two genuine remote
    matches. One more real, reproducible case where nomic is the safer
    choice, on an axis (remote preference) that -- unlike the language
    finding above -- this project's actual candidate cares about.
    """
    path = Path(__file__).resolve().parent / "realistic_cases.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(name=name, query=spec["query"],
             documents=[(text, rel) for text, rel in spec["documents"]])
        for name, spec in data.items()
    ]


@dataclass
class Row:
    model: str
    scenario: str
    precision: float
    recall: float
    ndcg: float


def run_case(retriever, case: Case, k: int | None = None) -> Row:
    docs = [text for text, _ in case.documents]
    labels = [rel for _, rel in case.documents]
    total_relevant = sum(labels)
    # These cases are small (3-5 documents) -- capping k at the document
    # count (what the golden-set path does; fine there since golden sets have
    # dozens of vacancies) would make p@k/recall@k degenerate: k >= len(docs)
    # means every document is "in the top k" regardless of ranking quality,
    # so precision collapses to the base rate and recall is always 1.0. Using
    # k = the relevant count instead ("precision at R") keeps both metrics
    # sensitive to whether relevant documents actually rank above irrelevant
    # ones, which is the entire point of a case like this.
    if k is None:
        k = max(1, total_relevant)
    ranked = retriever.top_n(case.query, docs, len(docs))
    ordered = [labels[i] for i, _ in ranked]
    return Row(
        model=retriever.name,
        scenario=f"case:{case.name}",
        precision=precision_at_k(ordered, k),
        recall=recall_at_k(ordered, k, total_relevant),
        ndcg=ndcg_at_k(ordered, k),
    )


def run_golden(retriever, golden: GoldenSet, k: int) -> Row:
    result = evaluate(golden, retriever, k=k)
    return Row(
        model=retriever.name,
        scenario=f"golden:{golden.name}",
        precision=result.precision,
        recall=result.recall,
        ndcg=result.ndcg,
    )


def print_table(rows: list[Row]) -> None:
    header = ("model", "scenario", "p@k", "recall@k", "nDCG@k")
    formatted = [
        (r.model, r.scenario, f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.ndcg:.3f}")
        for r in rows
    ]
    widths = [max(len(str(row[i])) for row in ([header] + formatted)) for i in range(5)]

    def line(row) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(row, widths))

    print(line(header))
    print("  ".join("-" * w for w in widths))
    for row in formatted:
        print(line(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="http://localhost:8080",
                         help="llama.cpp / llama-swap OpenAI-compatible host")
    parser.add_argument("--golden-dir", default="data/golden",
                         help="directory of golden set JSON files")
    parser.add_argument("--k", type=int, default=10, help="cutoff for p@k/recall@k/nDCG@k")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-bm25", action="store_true",
                         help="skip the model-independent bm25 baseline row")
    args = parser.parse_args()

    golden_sets = GoldenSet.load_dir(Path(args.golden_dir))
    cases = synthetic_cases() + realistic_cases()
    rows: list[Row] = []

    def case_k(case: Case) -> int:
        # See run_case's docstring comment: capping at len(documents) would
        # make p@k/recall@k degenerate for cases this small. --k still wins
        # if the caller explicitly wants a smaller cutoff than that.
        return min(args.k, max(1, sum(rel for _, rel in case.documents)))

    if not args.skip_bm25:
        bm25 = BM25Retriever()
        for case in cases:
            rows.append(run_case(bm25, case, case_k(case)))
        for golden in golden_sets:
            rows.append(run_golden(bm25, golden, min(args.k, len(golden.vacancies))))

    for model in DEFAULT_MODELS:
        client = LlamaCppClient(
            args.host,
            model="unused",  # chat model is never called by embed-only benchmarking
            embed_model=model.embed_model,
            timeout=args.timeout,
            embed_query_prefix=model.query_prefix,
            embed_document_prefix=model.document_prefix,
        )
        retriever = EmbeddingRetriever(client)
        retriever.name = model.label  # readable label instead of "embeddings"

        for case in cases:
            rows.append(run_case(retriever, case, case_k(case)))
        for golden in golden_sets:
            rows.append(run_golden(retriever, golden, min(args.k, len(golden.vacancies))))

    print_table(rows)


if __name__ == "__main__":
    main()
