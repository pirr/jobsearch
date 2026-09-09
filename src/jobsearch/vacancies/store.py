"""SQLite persistence -- the seam between ingest, rating, and presentation.

Stages never call each other; they meet here. That is what makes a crashed
ingest resumable, keeps one dead source from losing another's work, and stops
re-runs from re-paying for ratings that already exist.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .models import Vacancy

SCHEMA = """
CREATE TABLE IF NOT EXISTS vacancies (
    id               TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    source_id        TEXT NOT NULL,
    url              TEXT NOT NULL,
    company          TEXT NOT NULL,
    title            TEXT NOT NULL,
    location_raw     TEXT,
    location_city    TEXT,
    location_country TEXT,
    is_remote        INTEGER,
    employment_type  TEXT,
    description      TEXT NOT NULL,
    posted_at        TEXT,
    fetched_at       TEXT NOT NULL,
    raw_json         TEXT
);
CREATE INDEX IF NOT EXISTS idx_vacancies_source  ON vacancies(source);
CREATE INDEX IF NOT EXISTS idx_vacancies_posted  ON vacancies(posted_at);

CREATE TABLE IF NOT EXISTS ratings (
    vacancy_id     TEXT NOT NULL,
    profile_hash   TEXT NOT NULL,
    score          INTEGER,
    verdict        TEXT,
    matched_skills TEXT,
    missing_skills TEXT,
    evidence       TEXT,
    hard_blockers  TEXT,
    model          TEXT,
    failed         INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    rated_at       TEXT NOT NULL,
    PRIMARY KEY (vacancy_id, profile_hash),
    FOREIGN KEY (vacancy_id) REFERENCES vacancies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vacancy_status (
    vacancy_id TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (vacancy_id) REFERENCES vacancies(id) ON DELETE CASCADE
);

-- Manual escape hatch for what dedupe.py's hash-based identity misses: a
-- typo'd company name or a differently formatted location_raw across
-- sources produces two ids for what is really one posting. `vacancy_id` is
-- hidden from ranked(); `canonical_id` is the one that stays visible and is
-- never itself hidden by this table.
CREATE TABLE IF NOT EXISTS vacancy_duplicates (
    vacancy_id   TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    marked_at    TEXT NOT NULL,
    FOREIGN KEY (vacancy_id) REFERENCES vacancies(id) ON DELETE CASCADE,
    FOREIGN KEY (canonical_id) REFERENCES vacancies(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    ok          INTEGER,
    error       TEXT,
    count       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS profiles (
    id           TEXT PRIMARY KEY,
    profile_hash TEXT NOT NULL,
    source_files TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profiles_hash ON profiles(profile_hash);
"""

VALID_STATUSES = ("seen", "dismissed", "applied", "closed")

_SORT_COLUMNS = {
    "score": "r.score",
    "posted_at": "v.posted_at",
    "company": "v.company",
    "title": "v.title",
}
# posted_at is the only nullable sort column here (r.score is guaranteed
# non-null by the r.score >= :threshold filter below -- NULL >= number is
# NULL, i.e. excluded; company/title are NOT NULL in the schema). COALESCE
# it to a sentinel that always sorts after every real value in either
# direction, so unposted-date rows land at the end of the list regardless
# of sort_dir instead of SQLite's default rule (NULLs sort first in ASC),
# which would scatter them to the front.
_NULL_SENTINELS = {"desc": "", "asc": "9999-12-31T23:59:59+00:00"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- vacancies -----------------------------------------------------

    def upsert_vacancies(self, vacancies: Iterable[Vacancy]) -> tuple[int, int]:
        """Insert or refresh. Returns (new, updated).

        Idempotent on the dedupe key, so re-running ingest leaves the row count
        stable instead of growing a duplicate per run.
        """
        new = updated = 0
        for v in vacancies:
            existing = self.conn.execute(
                "SELECT 1 FROM vacancies WHERE id = ?", (v.id,)
            ).fetchone()
            self.conn.execute(
                """
                INSERT INTO vacancies (id, source, source_id, url, company, title,
                    location_raw, location_city, location_country, is_remote,
                    employment_type, description, posted_at, fetched_at, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    url=excluded.url,
                    description=excluded.description,
                    location_raw=excluded.location_raw,
                    location_city=excluded.location_city,
                    location_country=excluded.location_country,
                    is_remote=excluded.is_remote,
                    employment_type=excluded.employment_type,
                    posted_at=COALESCE(excluded.posted_at, vacancies.posted_at),
                    fetched_at=excluded.fetched_at,
                    raw_json=excluded.raw_json
                """,
                (
                    v.id, v.source, v.source_id, v.url, v.company, v.title,
                    v.location_raw, v.location_city, v.location_country,
                    None if v.is_remote is None else int(v.is_remote),
                    v.employment_type, v.description, _iso(v.posted_at), _now(),
                    json.dumps(v.raw, ensure_ascii=False),
                ),
            )
            if existing:
                updated += 1
            else:
                new += 1
        self.conn.commit()
        return new, updated

    def count_vacancies(self, source: str | None = None) -> int:
        if source:
            row = self.conn.execute(
                "SELECT COUNT(*) c FROM vacancies WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) c FROM vacancies").fetchone()
        return row["c"]

    def get_vacancy(self, vacancy_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)
        ).fetchone()

    def update_description(self, vacancy_id: str, description: str) -> None:
        """Replace a vacancy's stored description -- for pasting in the full
        text after manually clicking through to it (e.g. Adzuna's API only
        ever returns a ~500-char snippet), one vacancy at a time."""
        self.conn.execute(
            "UPDATE vacancies SET description = ? WHERE id = ?", (description, vacancy_id)
        )
        self.conn.commit()

    def all_vacancies(self, exclude_dismissed: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT v.* FROM vacancies v"
        if exclude_dismissed:
            sql += (
                " LEFT JOIN vacancy_status s ON s.vacancy_id = v.id"
                " WHERE s.status IS NULL OR s.status != 'dismissed'"
            )
        return list(self.conn.execute(sql))

    def get_vacancies(self, ids: Sequence[str]) -> list[sqlite3.Row]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return list(
            self.conn.execute(f"SELECT * FROM vacancies WHERE id IN ({marks})", tuple(ids))
        )

    def close_missing_vacancies(
        self, source: str, company: str, present_source_ids: set[str]
    ) -> int:
        """Mark vacancies closed if a fresh fetch no longer returns them.

        Scoped to one (source, company) at a time -- the caller (ingest) is
        the one that just fetched that company's current live listing and
        knows exactly which ids it contains. Skips anything already `applied`
        or `closed`: applying is a fact worth keeping even after a role
        closes, not something a later disappearance should silently erase.
        Returns how many were newly marked.
        """
        rows = self.conn.execute(
            """
            SELECT v.id, v.source_id FROM vacancies v
            LEFT JOIN vacancy_status s ON s.vacancy_id = v.id
            WHERE v.source = ? AND v.company = ?
              AND COALESCE(s.status, 'new') NOT IN ('applied', 'closed')
            """,
            (source, company),
        )
        gone = [r["id"] for r in rows if r["source_id"] not in present_source_ids]
        for vacancy_id in gone:
            self.set_status(vacancy_id, "closed")
        return len(gone)

    # ---- ratings -------------------------------------------------------

    def rated_ids(self, profile_hash: str) -> set[str]:
        """Already-rated vacancies for this profile, so match never re-pays."""
        rows = self.conn.execute(
            "SELECT vacancy_id FROM ratings WHERE profile_hash = ? AND failed = 0",
            (profile_hash,),
        )
        return {r["vacancy_id"] for r in rows}

    def save_rating(
        self,
        vacancy_id: str,
        profile_hash: str,
        model: str,
        *,
        score: int | None = None,
        verdict: str | None = None,
        matched_skills=None,
        missing_skills=None,
        evidence=None,
        hard_blockers=None,
        failed: bool = False,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ratings (vacancy_id, profile_hash, score, verdict,
                matched_skills, missing_skills, evidence, hard_blockers,
                model, failed, error, rated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(vacancy_id, profile_hash) DO UPDATE SET
                score=excluded.score, verdict=excluded.verdict,
                matched_skills=excluded.matched_skills,
                missing_skills=excluded.missing_skills,
                evidence=excluded.evidence,
                hard_blockers=excluded.hard_blockers,
                model=excluded.model, failed=excluded.failed,
                error=excluded.error, rated_at=excluded.rated_at
            """,
            (
                vacancy_id, profile_hash, score, verdict,
                json.dumps(matched_skills or [], ensure_ascii=False),
                json.dumps(missing_skills or [], ensure_ascii=False),
                json.dumps(evidence or {}, ensure_ascii=False),
                json.dumps(hard_blockers or [], ensure_ascii=False),
                model, int(failed), error, _now(),
            ),
        )
        self.conn.commit()

    def ranked(
        self, profile_hash: str, threshold: int, limit: int, *, only_status: str | None = None
    ) -> list[sqlite3.Row]:
        """Ranked vacancies for this profile.

        By default excludes `dismissed` and `closed` (the two statuses that
        mean "not a real option any more"). Pass `only_status` to flip that
        around and see just that one bucket instead -- e.g. `show --closed`.
        Also always excludes anything marked a duplicate (mark_duplicate) --
        unconditionally, even under `only_status`, since that is a fact about
        the data rather than a per-candidate judgment call like status is.
        """
        if only_status:
            status_clause = "COALESCE(s.status,'new') = ?"
            params = (profile_hash, threshold, only_status, limit)
        else:
            status_clause = "COALESCE(s.status,'new') NOT IN ('dismissed', 'closed')"
            params = (profile_hash, threshold, limit)
        return list(
            self.conn.execute(
                f"""
                SELECT v.*, r.score, r.verdict, r.matched_skills, r.missing_skills,
                       r.hard_blockers, r.model, r.evidence, COALESCE(s.status,'new') AS status
                FROM ratings r
                JOIN vacancies v ON v.id = r.vacancy_id
                LEFT JOIN vacancy_status s ON s.vacancy_id = v.id
                WHERE r.profile_hash = ? AND r.failed = 0 AND r.score >= ?
                  AND v.id NOT IN (SELECT vacancy_id FROM vacancy_duplicates)
                  AND {status_clause}
                ORDER BY r.score DESC
                LIMIT ?
                """,
                params,
            )
        )

    def ranked_page(
        self,
        profile_hash: str,
        threshold: int,
        *,
        only_status: str | None = None,
        sort_by: str = "score",
        sort_dir: str = "desc",
        after: tuple[object, str] | None = None,
        page_size: int = 50,
    ) -> list[sqlite3.Row]:
        """Keyset-paginated vacancy list for the TUI.

        Ordered by (sort_expr, id) with a seek predicate on `after` -- never
        OFFSET, which silently duplicates or skips rows across pages when a
        status change mid-browse shrinks the filtered result set out from
        under a numeric offset. `after` is the (row["_sort_key"], row["id"])
        of the last row already loaded by the caller; pass None for the
        first page. Each row carries its own `_sort_key` so the caller never
        has to reimplement the COALESCE-sentinel logic to build the next
        cursor. Named (`:name`) parameters, not positional `?`, because the
        WHERE clause is assembled dynamically from several optional pieces.
        """
        if sort_by not in _SORT_COLUMNS:
            raise ValueError(f"sort_by must be one of {sorted(_SORT_COLUMNS)}, got {sort_by!r}")
        if sort_dir not in ("asc", "desc"):
            raise ValueError(f"sort_dir must be 'asc' or 'desc', got {sort_dir!r}")

        params: dict[str, object] = {
            "profile_hash": profile_hash,
            "threshold": threshold,
            "page_size": page_size,
        }
        sort_col = _SORT_COLUMNS[sort_by]
        if sort_by == "posted_at":
            sort_expr = f"COALESCE({sort_col}, :null_sentinel)"
            params["null_sentinel"] = _NULL_SENTINELS[sort_dir]
        else:
            sort_expr = sort_col

        if only_status:
            status_clause = "COALESCE(s.status,'new') = :status"
            params["status"] = only_status
        else:
            status_clause = "COALESCE(s.status,'new') NOT IN ('dismissed', 'closed')"

        where = [
            "r.profile_hash = :profile_hash",
            "r.failed = 0",
            "r.score >= :threshold",
            "v.id NOT IN (SELECT vacancy_id FROM vacancy_duplicates)",
            status_clause,
        ]
        if after is not None:
            after_value, after_id = after
            op = "<" if sort_dir == "desc" else ">"
            where.append(
                f"({sort_expr} {op} :after_value "
                f"OR ({sort_expr} = :after_value AND v.id > :after_id))"
            )
            params["after_value"] = after_value
            params["after_id"] = after_id

        sql = f"""
            SELECT v.*, r.score, r.verdict, r.matched_skills, r.missing_skills,
                   r.hard_blockers, r.model, r.evidence,
                   COALESCE(s.status,'new') AS status,
                   {sort_expr} AS _sort_key
            FROM ratings r
            JOIN vacancies v ON v.id = r.vacancy_id
            LEFT JOIN vacancy_status s ON s.vacancy_id = v.id
            WHERE {" AND ".join(where)}
            ORDER BY {sort_expr} {sort_dir.upper()}, v.id ASC
            LIMIT :page_size
        """
        return list(self.conn.execute(sql, params))

    def failed_ratings(self, profile_hash: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM ratings WHERE profile_hash = ? AND failed = 1",
            (profile_hash,),
        ).fetchone()
        return row["c"]

    # ---- status --------------------------------------------------------

    def set_status(self, vacancy_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
        self.conn.execute(
            """INSERT INTO vacancy_status (vacancy_id, status, updated_at) VALUES (?,?,?)
               ON CONFLICT(vacancy_id) DO UPDATE SET
                   status=excluded.status, updated_at=excluded.updated_at""",
            (vacancy_id, status, _now()),
        )
        self.conn.commit()

    # ---- duplicates ------------------------------------------------------

    def mark_duplicate(self, vacancy_id: str, canonical_id: str) -> str | None:
        """Mark vacancy_id as a duplicate of canonical_id, hiding it from
        ranked(). If vacancy_id carries a status (seen/dismissed/applied)
        and canonical_id doesn't have one of its own yet, that status
        transfers -- otherwise a fact like "I applied to this" would
        silently vanish from view along with the now-hidden duplicate.

        Returns a warning string if vacancy_id's status conflicts with a
        status canonical_id already has of its own (never overwritten
        automatically -- the caller decides), otherwise None.
        """
        if vacancy_id == canonical_id:
            raise ValueError("a vacancy cannot be marked as a duplicate of itself")

        note = None
        dup_status = self.conn.execute(
            "SELECT status FROM vacancy_status WHERE vacancy_id = ?", (vacancy_id,)
        ).fetchone()
        if dup_status:
            canonical_status = self.conn.execute(
                "SELECT status FROM vacancy_status WHERE vacancy_id = ?", (canonical_id,)
            ).fetchone()
            if canonical_status is None:
                self.set_status(canonical_id, dup_status["status"])
            elif canonical_status["status"] != dup_status["status"]:
                note = (
                    f"{vacancy_id} was marked {dup_status['status']!r} but "
                    f"{canonical_id} already has status {canonical_status['status']!r} "
                    "-- not overwritten, review manually"
                )

        self.conn.execute(
            """INSERT INTO vacancy_duplicates (vacancy_id, canonical_id, marked_at)
               VALUES (?,?,?)
               ON CONFLICT(vacancy_id) DO UPDATE SET
                   canonical_id=excluded.canonical_id, marked_at=excluded.marked_at""",
            (vacancy_id, canonical_id, _now()),
        )
        self.conn.commit()
        return note

    def unmark_duplicate(self, vacancy_id: str) -> None:
        self.conn.execute("DELETE FROM vacancy_duplicates WHERE vacancy_id = ?", (vacancy_id,))
        self.conn.commit()

    # ---- ingest runs ---------------------------------------------------

    def start_run(self, source: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO ingest_runs (source, started_at) VALUES (?,?)", (source, _now())
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, ok: bool, count: int, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE ingest_runs SET finished_at=?, ok=?, count=?, error=? WHERE id=?",
            (_now(), int(ok), count, error, run_id),
        )
        self.conn.commit()

    def recent_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM ingest_runs ORDER BY id DESC LIMIT ?", (limit,)
            )
        )

    # ---- profiles -------------------------------------------------------

    def save_profile_record(
        self, profile_id: str, profile_hash: str, source_files: list[str]
    ) -> None:
        """Register an archived profile version so it shows up in `profiles`.

        Called after candidate.archive.archive_profile() has already copied
        the files to disk -- this just makes that archive discoverable.
        """
        self.conn.execute(
            "INSERT INTO profiles (id, profile_hash, source_files, created_at) "
            "VALUES (?,?,?,?)",
            (profile_id, profile_hash, json.dumps(source_files, ensure_ascii=False), _now()),
        )
        self.conn.commit()

    def get_profile(self, profile_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()

    def list_profiles(self) -> list[sqlite3.Row]:
        """Every archived profile version, newest first, each annotated with
        how many vacancies are already rated for it -- so `profiles` can show
        whether reactivating one re-pays LLM calls or returns instantly.
        """
        return list(
            self.conn.execute(
                """
                SELECT p.*, COUNT(r.vacancy_id) AS rated_count
                FROM profiles p
                LEFT JOIN ratings r
                    ON r.profile_hash = p.profile_hash AND r.failed = 0
                GROUP BY p.id
                ORDER BY p.created_at DESC
                """
            )
        )
