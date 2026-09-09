# Interactive terminal UI (`jobsearch tui`)

## Problem

`jobsearch show` prints a static Rich table/panels to stdout. Past ~30 full
panels a terminal's scrollback becomes the only way to review anything, there
is no way to act on a vacancy without copying its 32-char id and running a
separate `jobsearch mark`/`mark-duplicate`/`update-description` command, and
every view is capped by `--limit`. This spec adds an interactive, scrollable
terminal UI that removes those limits and lets every existing read/write
action (status, duplicate marking, description editing) happen in place.

Considered and rejected: a browser-based web UI. Rejected because it requires
a new long-running service (this project's `jobsearch` docker-compose service
is explicitly one-shot, `run --rm`, per `CLAUDE.md`), a new dependency stack
(FastAPI/Jinja2/etc.), and buys nothing this is a personal, single-user,
local-machine tool needs (no other device needs to reach it). A TUI fits the
existing one-shot, interactive `docker compose run --rm` pattern exactly and
reuses 100% of the existing `Store`-based domain code.

## Architecture

- New package `src/jobsearch/tui/`, alongside the existing `candidate/`,
  `matching/`, `vacancies/`, `cli/` packages.
- New dependency: `textual>=8.2` (current stable per PyPI as of this spec),
  added to `pyproject.toml`'s core `dependencies` (not optional — this
  becomes a primary interface, not an add-on). Built by the same team as
  `rich`, which is already a dependency; async-native, which fits the
  existing asyncio ingest/rating code; `DataTable` widget virtualizes
  rendering so large row counts are cheap to display.
- New CLI command `jobsearch tui`, added to `cli/main.py`:
  ```python
  @app.command()
  def tui():
      """Launch the interactive terminal UI."""
      from ..tui.app import JobsearchApp
      JobsearchApp().run()
  ```
  The import is local so `textual` is only loaded for this command, keeping
  every other command's startup unaffected.
- Existing commands (`show`, `mark`, `mark-duplicate`, `update-description`,
  etc.) are unchanged and stay available for scripting/piping.
- Docker: `docker compose run --rm jobsearch tui`. No docker-compose.yml
  change — `docker compose run` allocates a pseudo-TTY by default when
  invoked from an interactive shell, same as today. Verify during
  implementation that `TERM`/`COLORTERM` propagate correctly for full-color
  rendering; if not, add `environment: TERM: ${TERM:-xterm-256color}`.
- Like the CLI (see `cli/main.py`'s module docstring), the TUI only ever
  reads/writes through `Store` — no parallel data-access path, no business
  logic duplicated into the UI layer.

## Screens

### List screen (main)

- Filter bar (top): text search `Input`, a view selector (**active** /
  dismissed / closed / **duplicates**), a sort `Select` (**Score** / Date
  posted / Company / Title) with a direction-toggle button next to it (both
  reached by `Tab`-ing through the filter bar, not a bare global hotkey — same
  as the search `Input` itself, which is only reachable directly via `/`),
  and a Remote-only switch.
- `DataTable`: score, verdict, status, title, company, location — same
  columns as `jobsearch show`'s table today, minus the raw id/url columns
  (id is carried as the row key, not displayed).
- `Footer` (Textual built-in) showing active key bindings.
- Selecting a row and pressing `Enter` pushes the detail screen.

### Detail screen

- Everything `_panel()` in `cli/main.py` renders today: company, location,
  status, matched skills with CV/vacancy evidence, missing skills, hard
  blockers, soft-penalty notes.
- The vacancy URL is rendered as a clickable terminal hyperlink (Rich
  `[link=...]` markup) rather than shelled out to a browser — this works
  inside the Docker container (no browser present there) via the terminal
  emulator's own OSC-8 handling (Ctrl+click in iTerm2/kitty/wezterm/gnome-
  terminal etc. opens it on the host). No `webbrowser.open()` call needed.
- Same action bindings as the list screen apply here too (`s`/`d`/`a`/`m`/`e`).
- `Esc`/`q` returns to the list screen without re-fetching it.

### Duplicate picker (modal)

Opened by `m` on a highlighted list/detail row (that row becomes
`vacancy_id`, the one to hide):

1. Modal shows "Mark `<title>` (`<company>`) as duplicate of…" with a search
   `Input`.
2. Typing queries `Store.search_vacancies(query, limit=20)` (debounced),
   matching `title`/`company`/`location`/`id` via `LIKE` — across **all** vacancies,
   not just the currently loaded/filtered page, since the canonical vacancy
   may be unrated, below threshold, or simply not loaded yet. Pasting a full
   id directly also works, since id is one of the matched fields.
3. Arrow keys move through results, `Enter` selects one as `canonical_id`.
4. A short confirm step ("hide X as a duplicate of Y?") before committing —
   this hides a vacancy from the default view, so it isn't a single
   accidental keystroke.
5. On confirm: `store.mark_duplicate(vacancy_id, canonical_id)`. Any warning
   string it returns (status conflict between the two) is shown via
   `app.notify(...)`. The modal closes and the now-hidden row is removed
   from the current list locally (see "Local mutation, not re-fetch" below).

### Duplicates view

A fourth option on the view selector (alongside active/dismissed/closed).
Backed by a new `Store.list_duplicates(after=None, page_size=50)`:

```sql
SELECT d.vacancy_id, d.canonical_id, d.marked_at,
       v1.title AS dup_title, v1.company AS dup_company,
       v2.title AS canonical_title, v2.company AS canonical_company
FROM vacancy_duplicates d
JOIN vacancies v1 ON v1.id = d.vacancy_id
JOIN vacancies v2 ON v2.id = d.canonical_id
ORDER BY d.marked_at DESC, d.vacancy_id ASC
```

Lists "hidden vacancy → kept as" pairs. An unmark action (bound to the same
key CLI uses conceptually, e.g. `u`) calls `store.unmark_duplicate(vacancy_id)`
and removes the row from this view locally — this is how a duplicate marked
by mistake gets reversed without needing to remember/paste its id manually.

## Automatic vs. manual duplicate handling (context, not new behavior)

`vacancies/dedupe.py` already merges duplicates automatically, entirely at
`jobsearch ingest` time: `Vacancy.id` is a hash of normalized
`(company, title, location, description[:500])`, and `Store.upsert_vacancies`
does `INSERT ... ON CONFLICT(id) DO UPDATE`, so a re-fetched posting that
normalizes identically just updates the existing row. This is unconditional
and untouched by this spec — the TUI never calls `upsert_vacancies`.

The manual `vacancy_duplicates` mechanism (what this spec's duplicate picker
drives) exists only for near-misses the hash doesn't catch: a typo'd company
name, differently formatted location, or a description snippet that differs
in its first 500 characters across two sources' APIs for the same real role.
Two such postings are, and remain, two distinct rows/ids until a human links
them — nothing merges them automatically after the fact.

## Pagination

Keyset (seek), not `OFFSET`. `Store` gets:

```python
def ranked_page(
    self, profile_hash: str, threshold: int, *,
    only_status: str | None = None,   # None=active | 'dismissed' | 'closed'
    search: str | None = None,        # LIKE across title/company/location
    remote_only: bool = False,
    sort_by: str = "score",           # score | posted_at | company | title
    sort_dir: str = "desc",           # asc | desc
    after: tuple[object, str] | None = None,  # (sort_value, id) cursor
    page_size: int = 50,
) -> list[sqlite3.Row]: ...
```

`sort_by`/`sort_dir` are validated against an explicit allowlist mapping to
real column expressions (`{"score": "r.score", "posted_at": "v.posted_at",
"company": "v.company", "title": "v.title"}`) before being interpolated into
the query — never interpolated from raw text. `posted_at` can be `NULL`;
order with `NULLS LAST` for that column regardless of direction so unposted-
date rows don't scatter to the front.

Ordering is `(sort_column, id)` with the tiebreak keyset predicate
`sort_column < :cursor_value OR (sort_column = :cursor_value AND id > :cursor_id)`
(flipped comparison for `asc`). This is immune to the classic `OFFSET`
pagination bug: if a vacancy is marked `dismissed` mid-browse and drops out
of the active filter, every later row does *not* shift by one, so nothing
already shown reappears and nothing gets silently skipped on the next page.

**UI model: load-more, not page 1/2/3.** The `DataTable` starts with the
first `page_size` rows; scrolling near the bottom (or a bound key) fetches
the next batch via the cursor and appends it. Rows already loaded are never
re-fetched or renumbered — there is nothing to merge across page boundaries,
since each vacancy is fetched at most once per browsing session, and
revisiting earlier ones is just scrolling up in the same table.

Changing the view (active/dismissed/closed/duplicates), the search text, the
sort column/direction, or the remote-only toggle resets to a fresh cursor
(`after=None`) and clears the table — each is logically a different filtered/
ordered result set, re-sorting only the already-loaded subset client-side
would misrepresent what's actually in the database.

**Local mutation, not re-fetch, for in-place actions.** Marking a status or a
duplicate does not re-query the list:
- If the new status (or a duplicate mark) makes a row no longer match the
  current view's filter, the row is removed from the already-rendered
  `DataTable` directly (`remove_row`) — remaining rows shift up as a normal
  widget redraw, not a reload. Scroll position is otherwise undisturbed.
- If the row still matches the current filter (e.g. `seen` → `applied` while
  viewing "active"), only its status cell is updated in place.
- The pagination cursor is keyed on `(sort_value, id)` of the last *loaded*
  row, so removing an already-loaded row from the widget never touches it.

**Accepted edge case:** editing a vacancy's description mid-session re-rates
it and can change its `score`, which can change where it *would* sort. The
running cursor won't retroactively reorder what's already fetched, and a
not-yet-loaded row's new score could place it before or after where it would
otherwise have appeared. `r` (refresh) starts a clean session with a fresh
cursor if that matters; this is a rare, explicitly user-triggered action, not
routine browsing.

## `search_vacancies`

```python
def search_vacancies(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
    """Match title/company/location/id, ignoring rating/threshold/status —
    used by the duplicate picker, which must be able to point at any
    vacancy in the database, not just ones currently ranked/visible."""
```

## Key bindings

| Key | Action |
|---|---|
| `/` | focus search input |
| `↑`/`↓`, scroll | move selection / load more near bottom |
| `Enter` | open detail screen for selected row |
| `s` / `d` / `a` | mark seen / dismissed / applied |
| `m` | open duplicate picker for selected row |
| `u` | (duplicates view) unmark selected pair |
| `e` | edit description in `$EDITOR`, then re-rate |
| `c` | cycle view: active → dismissed → closed → duplicates |
| `r` | refresh (fresh cursor, re-query current view) |
| `Esc` / `q` | back / quit |

## Editing description → re-rate

`e` calls a new shared helper, `src/jobsearch/editing.py`:

```python
def open_in_editor(current: str, *, run=subprocess.run) -> str:
    """Write current to a tempfile, run $EDITOR on it, return the result.
    `run` is injectable so tests don't shell out to a real editor."""
```

This factors out the editor-opening half of `cli/main.py`'s
`_read_description_input` (the interactive-terminal branch), used by both
the CLI and the TUI. In the TUI, it runs inside `app.suspend()` (Textual's
context manager that releases the terminal to a child process and restores
the UI after), matching how the CLI already shells out to `$EDITOR`.

After the editor closes: `store.update_description(...)` runs immediately
(fast, local), then re-rating (`rate_and_save`, an LLM call that can take
minutes per `JOBSEARCH_LLM_TIMEOUT`) runs in a Textual background worker
(`@work`) so the UI stays responsive — the affected row shows a spinner/
"rating…" state in its score column until the worker finishes or fails.

`cli/main.py`'s `_llm_hint(config)` moves to `matching/llm/__init__.py` as a
shared `llm_hint(config)` so the TUI can show the same backend-specific
troubleshooting text (Ollama not running, wrong llama.cpp host, missing
Anthropic key) that the CLI already shows on `LLMError`.

## Error handling

- No profile at `config.profile_path`, or zero ratings for it: the app shows
  a message screen (mirrors CLI's `_fail`) with the equivalent
  `jobsearch profile`/`jobsearch match` instructions, instead of opening an
  empty/broken list.
- `LLMError` during re-rate: caught in the worker, shown via `app.notify(...,
  severity="error")` with `llm_hint(config)` appended; the row's score cell
  reverts to its last known value rather than staying stuck on "rating…".
- `mark_duplicate` raising `ValueError` (self-duplicate): caught in the
  modal, shown inline instead of propagating.

## Testing

- Extend `tests/test_store.py`: `ranked_page` keyset behavior (no
  duplicate/skip across two calls when a status changes between them),
  `sort_by`/`sort_dir` per column including `posted_at` NULL ordering,
  `remote_only`, `search_vacancies`, `list_duplicates`.
- New `tests/test_tui.py` using Textual's `App.run_test()` pilot (fits the
  project's existing `pytest-asyncio`, `asyncio_mode = auto`): list renders
  from a temp `Store` with fixture vacancies, `s`/`d`/`a` update status and
  remove/keep the row per current view, `/` search narrows results, `m` →
  duplicate picker → confirm hides a row, detail screen opens/closes,
  load-more fetches the next batch at the scroll boundary. LLM re-rate paths
  use the existing `FakeLLM` (`matching/llm/fake.py`); `open_in_editor` is
  tested with an injected fake `run` instead of a real `$EDITOR` subprocess.

## Out of scope for v1

- Triggering `ingest`/`match` from within the TUI (long-running, network/LLM
  heavy) — stays CLI-only; run them before opening `jobsearch tui`.
- Any new docker-compose service, port, or auth — the TUI runs through the
  existing one-shot `jobsearch` service, interactively, exactly like every
  other command.

## Documentation (implementation-time task)

Add a `jobsearch tui` section to `README.md`'s Usage block, including the
`docker compose run --rm jobsearch tui` invocation.
