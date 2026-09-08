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
    hand-labelled vacancies. This is where the signal actually is: a live
    nomic-embed-text-v1.5 vs. embeddinggemma-300m run showed a real,
    consistent gap here (nDCG 1.000 vs 0.795) that traced back to
    embeddinggemma-300m over-weighting seniority/corporate register over
    actual domain match. Add more golden sets for broader real-world
    coverage; `GoldenSet.load_dir` already picks up every file there.
  - Synthetic cases (below) -- small, fast, hand-built probes. They were
    written to reproduce specific failure modes found in that same golden-set
    comparison, but running them revealed a limitation worth knowing before
    trusting them: short, clean, hand-written sentences don't carry the noise
    real job postings do, and two of the three ("jargon-synonyms",
    "seniority-vs-domain") turned out too easy -- every model, even bm25,
    aces them, so they currently confirm baseline competence rather than
    differentiate models. Only "abbreviation-expansion" actually separates
    the two models here (and in the opposite direction from the golden-set
    result: embeddinggemma-300m gets it, nomic-embed-text-v1.5 doesn't).
    Treat the synthetic cases as a fast smoke test, not a substitute for
    golden-set evidence -- if you want them to actually stress models the way
    the golden set does, they need harder, noisier distractors closer to real
    posting text, not cleaner ones.

Usage:
    python scripts/benchmark_embeddings.py
    python scripts/benchmark_embeddings.py --host http://localhost:8080 --k 5
"""

from __future__ import annotations

import argparse
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
            query="ML Engineer with NLP experience",
            documents=[
                ("Machine Learning Engineer, natural language processing, "
                 "transformers", True),
                ("Data Engineer, ETL pipelines, no ML experience required", False),
                ("Sales Manager, enterprise software, B2B", False),
            ],
        ),
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
    parser.add_argument("--host", default="http://192.168.1.20:8080",
                         help="llama.cpp / llama-swap OpenAI-compatible host")
    parser.add_argument("--golden-dir", default="data/golden",
                         help="directory of golden set JSON files")
    parser.add_argument("--k", type=int, default=10, help="cutoff for p@k/recall@k/nDCG@k")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-bm25", action="store_true",
                         help="skip the model-independent bm25 baseline row")
    args = parser.parse_args()

    golden_sets = GoldenSet.load_dir(Path(args.golden_dir))
    cases = synthetic_cases()
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
