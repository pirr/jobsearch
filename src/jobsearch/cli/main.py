"""Terminal frontend.

Every command reads or writes SQLite and nothing else, so the domain code never
prints. That is what keeps a web frontend additive later rather than a rewrite.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..candidate.archive import archive_profile
from ..candidate.extractor import extract_profile
from ..candidate.factory import ReaderFactory
from ..candidate.profile import CandidateProfile
from ..config import load_config
from ..matching.evaluation import GoldenSet, evaluate
from ..matching.llm import build_embedder, build_llm
from ..matching.llm.base import LLMError
from ..matching.pipeline import run_match
from ..matching.retrieval import BM25Retriever, build_retriever
from ..vacancies.ingest import build_sources, check_sources, ingest
from ..vacancies.store import Store

app = typer.Typer(add_completion=False, help="Rank job vacancies against your CV.")
console = Console()


def _store() -> Store:
    return Store(load_config().db_path)


def _fail(message: str) -> None:
    console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(1)


def _llm_hint(config) -> str:
    """Backend-appropriate troubleshooting line for an LLMError.

    The three backends fail in different ways for different reasons, so a
    hint naming the wrong one (e.g. always suggesting `ollama serve` while
    JOBSEARCH_LLM=llamacpp) sends someone chasing the wrong server.
    """
    backend = config.llm_backend.lower()
    if backend == "ollama":
        return f"Is Ollama running? Try: ollama serve && ollama pull {config.ollama_model}"
    if backend == "llamacpp":
        return (
            f"Is llama-server running and reachable at {config.llamacpp_host}? "
            "If it's on another machine, check the port and that it's bound to "
            "0.0.0.0 rather than only 127.0.0.1."
        )
    if backend == "anthropic":
        return "Check ANTHROPIC_API_KEY is set and valid."
    return ""


def _fail_llm(exc: Exception, config) -> None:
    hint = _llm_hint(config)
    _fail(f"{exc}\n\n{hint}" if hint else str(exc))


@app.command()
def profile(
    files: list[Path] = typer.Argument(..., help="CV and any supporting documents"),
    out: Path = typer.Option(None, help="Where to write the profile JSON"),
):
    """Read documents and extract a candidate profile."""
    config = load_config()
    out = out or config.profile_path
    factory = ReaderFactory()
    try:
        text = factory.read_all(files)
    except Exception as exc:
        _fail(str(exc))
    console.print(f"read {len(files)} document(s), {len(text):,} characters")

    try:
        llm = build_llm(config)
        with console.status(f"extracting profile via {llm.name}..."):
            result = extract_profile(llm, text, [str(f) for f in files])
    except LLMError as exc:
        _fail_llm(exc, config)
    except Exception as exc:
        _fail(str(exc))

    result.save(out)
    console.print(f"[green]wrote[/] {out}  (profile hash {result.hash()})")
    try:
        profile_id, _dest_dir = archive_profile(config.profile_storage_path, result)
        with _store() as store:
            store.save_profile_record(
                profile_id, result.hash(), [Path(p).name for p in result.source_files]
            )
        console.print(f"[dim]archived as {profile_id}[/]")
    except Exception as exc:
        # profile.json above is the critical path match/show depend on;
        # a disk/permissions problem archiving must not block that.
        console.print(f"[yellow]warning:[/] could not archive this profile version: {exc}")
    console.print(f"  skills: {', '.join(result.skills[:12]) or '(none found)'}")
    if result.hard_constraints:
        console.print("  [bold]hard filters[/] (stated in your CV, these exclude vacancies):")
        for c in result.hard_constraints:
            console.print(f"    - {c.kind}: {c.value}   [dim]“{c.evidence[:60]}”[/]")
    if result.soft_constraints:
        console.print("  [dim]soft signals (inferred -- they only adjust score):[/]")
        for c in result.soft_constraints:
            console.print(f"    [dim]- {c.kind}: {c.value}[/]")
    console.print(f"\n[dim]Edit {out} if anything is wrong; match re-rates when it changes.[/]")


def ingest_cmd(
    source: str = typer.Option(
        None, "--source", "-s",
        help="arbeitnow|remoteok|jobicy|greenhouse|lever|ashby",
    ),
    company: str = typer.Option(None, "--company", "-c", help="Single ATS company slug"),
):
    """Fetch vacancies from sources into the local database."""
    config = load_config()
    try:
        sources = build_sources(config, only=source, company=company)
    except ValueError as exc:
        _fail(str(exc))

    console.print(f"ingesting from {len(sources)} source(s)...")
    with _store() as store:
        before = store.count_vacancies()

        def show(result):
            if result.ok:
                console.print(f"  [green]ok[/]   {result.label:28} {result.fetched:4} fetched, {result.new:4} new")
            else:
                console.print(f"  [red]fail[/] {result.label:28} {result.error[:70]}")

        results = asyncio.run(ingest(config, store, sources, progress=show))
        after = store.count_vacancies()

    ok = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    console.print(
        f"\n[bold]{ok}/{len(results)} sources ok[/] -- "
        f"{after - before} new, {after:,} vacancies stored"
    )
    if failed:
        console.print(f"[yellow]{len(failed)} source(s) failed[/] (run `jobsearch doctor` for detail)")


@app.command()
def match(
    force: bool = typer.Option(False, help="Re-rate vacancies already rated for this profile"),
    shortlist: int = typer.Option(None, help="How many vacancies reach the LLM"),
    retriever: str = typer.Option(None, help="bm25|embeddings"),
):
    """Filter, shortlist, and rate vacancies against the profile."""
    config = load_config()
    if not config.profile_path.exists():
        _fail(f"no profile at {config.profile_path} -- run `jobsearch profile <cv.pdf>` first")
    candidate = CandidateProfile.load(config.profile_path)

    try:
        llm = build_llm(config)
        name = retriever or config.retriever
        engine = build_retriever(name, build_embedder(config) if name == "embeddings" else None)
    except LLMError as exc:
        _fail_llm(exc, config)
    except ValueError as exc:
        _fail(str(exc))

    size = shortlist or config.shortlist_size
    with _store() as store:
        if store.count_vacancies() == 0:
            _fail("no vacancies stored -- run `jobsearch ingest` first")

        console.print(
            f"profile {candidate.hash()} | retriever {engine.name} | "
            f"rater {llm.name} | shortlist {size}"
        )
        done = [0]

        def show(row, outcome):
            done[0] += 1
            mark = "[red]x[/]" if not outcome.ok else f"{outcome.rating.score:3}"
            console.print(f"  [{done[0]:3}] {mark}  {row['title'][:56]}")

        try:
            report = run_match(
                store, candidate, llm, engine,
                shortlist_size=size, force=force, progress=show,
            )
        except LLMError as exc:
            _fail_llm(exc, config)

    console.print(
        f"\n{report.total:,} stored -> {report.after_filters:,} after hard filters "
        f"-> {report.shortlisted} shortlisted"
    )
    console.print(
        f"LLM calls: [bold]{report.llm_calls}[/] "
        f"(skipped {report.already_rated} already rated) | "
        f"rated {report.rated}, failed {report.failed}"
    )
    if report.excluded:
        console.print(f"[dim]{len(report.excluded)} excluded by hard filters, e.g. {report.excluded[0][1]}[/]")
    if report.dropped_claims:
        console.print(
            f"[yellow]{report.dropped_claims} unsupported claim(s) dropped[/] "
            "[dim](model quoted text not present in the sources)[/]"
        )
    console.print("\nrun [bold]jobsearch show[/] to see the ranking")


@app.command()
def show(
    limit: int = typer.Option(20, help="How many to display"),
    threshold: int = typer.Option(None, help="Minimum score"),
    full: bool = typer.Option(False, help="Show reasons and evidence"),
    sort: str = typer.Option(
        "asc", "--sort",
        help="asc|desc -- score order. asc ends with the best match, "
             "the part visible without scrolling up in a terminal.",
    ),
):
    """Show ranked vacancies."""
    if sort not in ("asc", "desc"):
        _fail(f"--sort must be asc or desc, got {sort!r}")

    config = load_config()
    if not config.profile_path.exists():
        _fail(f"no profile at {config.profile_path}")
    candidate = CandidateProfile.load(config.profile_path)
    cutoff = config.score_threshold if threshold is None else threshold

    with _store() as store:
        rows = store.ranked(candidate.hash(), cutoff, limit)  # score DESC, best first
        failed = store.failed_ratings(candidate.hash())

    if not rows:
        console.print(f"nothing scored at or above {cutoff}. Try --threshold 40, or `jobsearch match`.")
        raise typer.Exit(0)

    if sort == "asc":
        rows = list(reversed(rows))

    if full:
        for row in rows:
            _panel(row)
    else:
        table = Table(show_lines=True)
        table.add_column("score", justify="right", style="bold")
        table.add_column("verdict")
        table.add_column("status")
        table.add_column("title", max_width=44)
        table.add_column("company", max_width=18)
        table.add_column("location", max_width=16)
        # Full 32-char id, not truncated: Store.set_status matches it exactly
        # (no short-hash support), so a truncated id couldn't be pasted
        # straight into `jobsearch mark <id> ...`.
        table.add_column("id", no_wrap=True)
        # no_wrap, not a max_width + fold: folding breaks a URL across
        # multiple rendered lines with real line breaks, which corrupts both
        # copy-paste and a terminal's own link auto-detection. no_wrap keeps
        # it one unbroken string and lets the terminal soft-wrap the display
        # instead, which neither of those care about.
        table.add_column("link", no_wrap=True)
        for row in rows:
            colour = "green" if row["score"] >= 80 else "yellow" if row["score"] >= 60 else "white"
            status = row["status"]
            status_text = f"[cyan]{status}[/]" if status == "applied" else f"[dim]{status}[/]"
            table.add_row(
                f"[{colour}]{row['score']}[/]", row["verdict"], status_text, row["title"],
                row["company"], row["location_raw"] or "-", row["id"], row["url"],
            )
        console.print(table)
        console.print("[dim]`jobsearch show --full` for why each fits, `jobsearch mark <id> dismissed` to hide one.[/]")
    if failed:
        console.print(f"[yellow]{failed} vacancy(s) could not be rated[/] [dim](model output failed validation)[/]")


def _panel(row) -> None:
    matched = json.loads(row["matched_skills"] or "[]")
    missing = json.loads(row["missing_skills"] or "[]")
    blockers = json.loads(row["hard_blockers"] or "[]")
    status = row["status"]
    status_text = f"[cyan]status: {status}[/]" if status == "applied" else f"[dim]status: {status}[/]"
    body = [
        f"[bold]{row['company']}[/] · {row['location_raw'] or 'location unspecified'}",
        row["url"], status_text, "",
    ]
    if matched:
        body.append("[green]fits:[/]")
        for m in matched[:6]:
            body.append(f"  ✓ {m.get('skill','')}")
            body.append(f"    [dim]CV: “{m.get('evidence_cv','')[:88]}”[/]")
            body.append(f"    [dim]job: “{m.get('evidence_vacancy','')[:88]}”[/]")
    if missing:
        body.append(f"[yellow]missing:[/] {', '.join(missing[:8])}")
    if blockers:
        body.append(f"[red]blockers:[/] {'; '.join(blockers[:4])}")
    console.print(Panel("\n".join(body), title=f"[bold]{row['score']}[/] · {row['title']}",
                        subtitle=f"[dim]{row['id']}[/]", title_align="left"))


@app.command()
def mark(vacancy_id: str, status: str = typer.Argument(..., help="seen|dismissed|applied")):
    """Record that you have seen, dismissed, or applied to a vacancy."""
    with _store() as store:
        try:
            store.set_status(vacancy_id, status)
        except ValueError as exc:
            _fail(str(exc))
    console.print(f"[green]{vacancy_id}[/] -> {status}")


@app.command()
def profiles():
    """List every archived profile version."""
    config = load_config()
    active_hash = None
    if config.profile_path.exists():
        try:
            active_hash = CandidateProfile.load(config.profile_path).hash()
        except Exception:
            active_hash = None

    with _store() as store:
        rows = store.list_profiles()

    if not rows:
        console.print("no archived profiles yet -- run `jobsearch profile <cv.pdf>`")
        raise typer.Exit(0)

    table = Table()
    for col in ("", "id", "created", "profile hash", "source files", "rated"):
        table.add_column(col)
    for r in rows:
        mark_active = "[green]*[/]" if r["profile_hash"] == active_hash else ""
        sources = ", ".join(json.loads(r["source_files"] or "[]"))
        table.add_row(
            mark_active, r["id"], (r["created_at"] or "")[:19],
            r["profile_hash"], sources, str(r["rated_count"]),
        )
    console.print(table)
    console.print("[dim]`jobsearch use-profile <id>` to reactivate one.[/]")


@app.command(name="use-profile")
def use_profile(profile_id: str = typer.Argument(..., help="id from `jobsearch profiles`")):
    """Copy an archived profile back onto the active profile path."""
    config = load_config()
    with _store() as store:
        record = store.get_profile(profile_id)
    if record is None:
        _fail(f"no archived profile with id {profile_id!r} -- run `jobsearch profiles` to list them")

    archived_path = config.profile_storage_path / profile_id / "profile.json"
    if not archived_path.exists():
        _fail(f"archive record exists but {archived_path} is missing on disk")
    try:
        candidate = CandidateProfile.load(archived_path)
    except Exception as exc:
        _fail(f"archived profile.json is unreadable: {exc}")

    candidate.save(config.profile_path)
    console.print(f"[green]{config.profile_path}[/] <- {profile_id}  (profile hash {candidate.hash()})")

    with _store() as store:
        rated = len(store.rated_ids(candidate.hash()))
    if rated:
        console.print(f"[dim]{rated} vacancy(s) already rated -- `jobsearch show` works with no `match` first.[/]")
    else:
        console.print("[dim]no ratings yet for this profile -- run `jobsearch match`.[/]")


@app.command()
def doctor():
    """Check every configured source, and report dead company slugs."""
    config = load_config()
    sources = build_sources(config)
    console.print(f"checking {len(sources)} source(s)...")
    results = asyncio.run(check_sources(config, sources))
    live = [r for r in results if r.ok and r.fetched]
    empty = [r for r in results if r.ok and not r.fetched]
    dead = [r for r in results if not r.ok]

    console.print(f"\n[green]{len(live)} live[/] · [yellow]{len(empty)} empty[/] · [red]{len(dead)} dead[/]")
    for r in sorted(live, key=lambda r: -r.fetched)[:50]:
        console.print(f"  [green]ok[/]    {r.label:28} {r.fetched:5} postings")
    for r in empty:
        console.print(f"  [yellow]empty[/] {r.label:28} responded but has no postings")
    for r in dead:
        console.print(f"  [red]dead[/]  {r.label:28} {r.error[:64]}")
    if dead:
        console.print(f"\n[dim]Remove dead slugs from {config.companies_path}.[/]")


@app.command()
def runs(limit: int = 20):
    """Show recent ingest runs."""
    with _store() as store:
        rows = store.recent_runs(limit)
    table = Table()
    for col in ("source", "started", "ok", "count", "error"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["source"], (r["started_at"] or "")[:19],
                      "-" if r["ok"] is None else ("yes" if r["ok"] else "no"),
                      str(r["count"]), (r["error"] or "")[:44])
    console.print(table)


def eval_cmd(
    directory: Path = typer.Option(Path("data/golden"), "--dir"),
    k: int = typer.Option(10),
    retrievers: str = typer.Option("bm25", help="Comma-separated: bm25,embeddings"),
):
    """Score retrieval quality against the golden set."""
    sets = GoldenSet.load_dir(directory)
    if not sets:
        _fail(f"no golden files in {directory}")
    config = load_config()

    table = Table(title=f"retrieval quality @{k}")
    for col in ("golden set", "retriever", "p@k", "recall@k", "nDCG@k", "n", "relevant"):
        table.add_column(col)
    for name in [r.strip() for r in retrievers.split(",") if r.strip()]:
        try:
            engine = BM25Retriever() if name == "bm25" else build_retriever(name, build_embedder(config))
        except LLMError as exc:
            _fail_llm(exc, config)
        except ValueError as exc:
            _fail(str(exc))
        for golden in sets:
            try:
                r = evaluate(golden, engine, k=k)
            except LLMError as exc:
                console.print(f"[red]{name} failed:[/] {exc}\n{_llm_hint(config)}")
                break
            table.add_row(r.golden, r.retriever, f"{r.precision:.3f}",
                          f"{r.recall:.3f}", f"{r.ndcg:.3f}",
                          str(r.n_vacancies), str(r.n_relevant))
    console.print(table)


# Typer derives command names from functions; these two would otherwise collide
# with the builtins `ingest` and `eval`.
app.command(name="ingest")(ingest_cmd)
app.command(name="eval")(eval_cmd)


if __name__ == "__main__":
    app()
