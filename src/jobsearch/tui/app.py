"""The interactive terminal UI's main app: filter bar + vacancy list.

Like cli/main.py, this only ever reads/writes through Store -- no parallel
data-access path, no business logic duplicated into the UI layer.
"""

from __future__ import annotations

import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Middle, Center
from textual.widgets import Button, DataTable, Footer, Header, Input, Select, Static, Switch, Label

from .profile_modal import ProfilePickerModal
from ..candidate.profile import CandidateProfile
from ..config import load_config
from ..editing import open_in_editor
from ..matching.llm import build_llm, llm_hint
from ..matching.llm.base import LLMError
from ..matching.pipeline import rate_and_save
from ..vacancies.store import Store
from .duplicate_modal import DuplicatePickerModal
from .screens import DetailScreen, NoProfileScreen

_VIEWS = ["active", "new", "interview", "applied", "dismissed", "closed", "duplicates"]
_SORT_OPTIONS = [
    ("Score", "score"), ("Date posted", "posted_at"),
    ("Company", "company"), ("Title", "title"),
]
# Six separate footer entries for seen/dismissed/applied/interview/accepted/
# rejected don't fit an 80-column terminal (a common default, and Docker's
# fallback when no TTY size is negotiated) -- grouping them collapses each
# down to just its key, under one shared "Status" label, so they and the
# rest of the footer (notably the reason key) stay visible.
_STATUS_GROUP = Binding.Group("Status", compact=True)
_SEARCH_GROUP = Binding.Group("Search", compact=True)


class JobsearchApp(App):
    """Browse, filter and act on rated vacancies without a display limit."""

    CSS = """
    #topbar { height: 3; }
    #topbar Input { width: 1fr; }
    #topbar Select { width: 22; }
    #topbar Static { width: auto; }
    #remote-only-group { width: auto; }
    #remote-only { border: none; }
    """
    TITLE = "jobsearch"

    BINDINGS = [
        Binding("slash", "focus_search", "Search"),
        Binding("s", "mark_status('seen')", "Seen", group=_STATUS_GROUP),
        Binding("d", "mark_status('dismissed')", "Dismissed", group=_STATUS_GROUP),
        Binding("a", "mark_status('applied')", "Applied", group=_STATUS_GROUP),
        Binding("i", "mark_status('interview')", "Interview", group=_STATUS_GROUP),
        Binding("o", "mark_status('accepted')", "Accepted", group=_STATUS_GROUP),
        Binding("x", "mark_status('rejected')", "Rejected", group=_STATUS_GROUP),
        Binding("m", "mark_duplicate", "Duplicate"),
        Binding("u", "unmark_duplicate", "Unmark", show=False),
        Binding("e", "edit_description", "Edit desc"),
        Binding("n", "edit_reason", "Reason"),
        Binding("c", "cycle_view", "Cycle view"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "cancel_search", "Cancel search", show=False, group=_SEARCH_GROUP),
    ]

    def __init__(self, *, config=None, store=None, candidate=None, llm=None) -> None:
        super().__init__()
        self._config = config or load_config()
        self.store = store or Store(self._config.db_path)
        self._llm_override = llm
        self._candidate_override = candidate
        self.candidate: CandidateProfile | None = None

        self.view = "active"
        self.search_text = ""
        self.remote_only = False
        self.sort_by = "score"
        self.sort_dir = "desc"
        self.cursor = None
        self.page_size = 50
        self._loaded_ids: set[str] = set()
        self._row_cache: dict[str, object] = {}
        self._columns = []
        self._selected_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="topbar"):
            yield Input(placeholder="press / to search title/company/location", id="search")
            yield Select(_SORT_OPTIONS, value="score", id="sort", allow_blank=False)
            yield Button("↓ desc", id="sort-dir")

            with Vertical(id="remote-only-group"):
                yield Static("remote only", id="remote-only-label")
                yield Switch(id="remote-only")

            yield Button(id="profiles-btn", action="app.profiles_select")
            yield Label(id="rating-lbl")

        yield DataTable(id="table", cursor_type="row")
        yield Footer()

    async def on_mount(self) -> None:
        self.candidate = self._candidate_override
        if self.candidate is None:
            if not self._config.profile_path.exists():
                await self.push_screen(NoProfileScreen(
                    f"no profile at {self._config.profile_path} -- "
                    "run `jobsearch profile <cv.pdf>` first"))
                return
            try:
                self.candidate = CandidateProfile.load(self._config.profile_path)
            except Exception as exc:
                await self.push_screen(NoProfileScreen(f"profile.json is unreadable: {exc}"))
                return
        if not self.store.rated_ids(self.candidate.hash()):
            await self.push_screen(NoProfileScreen(
                "no rated vacancies for this profile -- run `jobsearch match` first"))
            return
        self.load_first_page()
        # Explicit, not left to Textual's AUTO_FOCUS heuristic -- reported in
        # the wild as an unfocused table (blurred/gray cursor row, hotkeys
        # needing a mouse click first) even though this same heuristic
        # focuses it fine under the headless test driver.
        self.query_one("#table", DataTable).focus()
        if self.candidate and self.candidate.source_files:
            self.query_one("#profiles-btn", Button).label = self.candidate.source_files[0]
            self.query_one("#rating-lbl", Label).update(self._get_last_rated_at())

    def _setup_columns_for_view(self) -> None:
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)
        if self.view == "duplicates":
            self._columns = table.add_columns("Marked", "Duplicate", "Canonical")
        else:
            self._columns = table.add_columns(
                "Score", "Verdict", "Status", "Title", "Company", "Location"
            )

    def load_first_page(self) -> None:
        self.cursor = None
        self._loaded_ids = set()
        self._setup_columns_for_view()
        self.load_more()

    def load_more(self) -> None:
        # Widgets in the filter bar (Select, Switch, ...) can post their
        # initial Changed message as soon as they mount, which bubbles up
        # to the App regardless of which screen is active -- including
        # while NoProfileScreen is showing and on_mount's guard already
        # returned early without ever setting self.candidate.
        if self.candidate is None and self.view != "duplicates":
            return

        table = self.query_one("#table", DataTable)

        if self.view == "duplicates":
            rows = self.store.list_duplicates(after=self.cursor, page_size=self.page_size)
            for r in rows:
                table.add_row(
                    r["marked_at"][:19],
                    f"{r['dup_title']} ({r['dup_company']})",
                    f"{r['canonical_title']} ({r['canonical_company']})",
                    key=r["vacancy_id"],
                )
                self._loaded_ids.add(r["vacancy_id"])
                self._row_cache[r["vacancy_id"]] = r
            if rows:
                self.cursor = (rows[-1]["marked_at"], rows[-1]["vacancy_id"])
            return

        rows = self.store.ranked_page(
            self.candidate.hash(), self._config.score_threshold,
            only_status=None if self.view == "active" else self.view,
            search=self.search_text or None,
            remote_only=self.remote_only,
            sort_by=self.sort_by, sort_dir=self.sort_dir,
            after=self.cursor, page_size=self.page_size,
        )
        for r in rows:
            table.add_row(
                str(r["score"]), r["verdict"] or "", r["status"],
                r["title"], r["company"], r["location_raw"] or "-",
                key=r["id"],
            )
            self._loaded_ids.add(r["id"])
            self._row_cache[r["id"]] = r
        if rows:
            self.cursor = (rows[-1]["_sort_key"], rows[-1]["id"])

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        self.search_text = event.value.strip()
        self.load_first_page()

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sort":
            return
        self.sort_by = event.value
        self.load_first_page()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "sort-dir":
            return
        self.sort_dir = "asc" if self.sort_dir == "desc" else "desc"
        event.button.label = "↑ asc" if self.sort_dir == "asc" else "↓ desc"
        self.load_first_page()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id != "remote-only":
            return
        self.remote_only = event.value
        self.load_first_page()

    def action_cycle_view(self) -> None:
        idx = _VIEWS.index(self.view)
        self.view = _VIEWS[(idx + 1) % len(_VIEWS)]
        self.load_first_page()

    def action_refresh(self) -> None:
        self.load_first_page()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._selected_id = str(event.row_key.value) if event.row_key.value else None
        table = self.query_one("#table", DataTable)
        if event.cursor_row >= table.row_count - 5:
            self.load_more()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self.view == "duplicates" or not event.row_key.value:
            return
        row = self._row_cache.get(str(event.row_key.value))
        if row is not None:
            self.push_screen(DetailScreen(row))

    def action_mark_status(self, status: str) -> None:
        if self.view == "duplicates" or self._selected_id is None:
            return
        self.mark_status_for_id(self._selected_id, status)

    def action_mark_duplicate(self) -> None:
        if self.view == "duplicates" or self._selected_id is None:
            return
        row = self._row_cache.get(self._selected_id)
        if row is None:
            return
        self.open_duplicate_picker(self._selected_id, row["title"], row["company"])

    def open_duplicate_picker(self, vacancy_id: str, title: str, company: str) -> None:
        def handle_result(canonical_id: str | None) -> None:
            if canonical_id is None:
                return
            try:
                note = self.store.mark_duplicate(vacancy_id, canonical_id)
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return
            if note:
                self.notify(note, severity="warning")
            if vacancy_id in self._loaded_ids:
                self.query_one("#table", DataTable).remove_row(vacancy_id)
                self._loaded_ids.discard(vacancy_id)

        self.push_screen(DuplicatePickerModal(self.store, vacancy_id, title, company), handle_result)

    def action_edit_description(self) -> None:
        if self.view == "duplicates" or self._selected_id is None:
            return
        self.edit_description_for_id(self._selected_id)

    def action_profiles_select(self) -> None:
        def handle_result(profile_id: str | None) -> None:
            if profile_id is None:
                return

            config = load_config()
            archived_path = config.profile_storage_path / profile_id / "profile.json"
            self.candidate = CandidateProfile.load(archived_path)
            self.candidate.save(config.profile_path)
            self.load_first_page()
            self.query_one("#table", DataTable).focus()
            if self.candidate and self.candidate.source_files:
                self.query_one("#profiles-btn", Button).label = self.candidate.source_files[0]
                self.query_one("#rating-lbl", Label).update(self._get_last_rated_at())

        self.push_screen(ProfilePickerModal(self.store), handle_result)

    def edit_description_for_id(self, vacancy_id: str) -> None:
        current = self.store.get_vacancy(vacancy_id)["description"] or ""
        new_text = self._prompt_editor(current)
        if not new_text or new_text == current:
            return
        self.store.update_description(vacancy_id, new_text)
        if vacancy_id in self._row_cache:
            self._row_cache[vacancy_id] = dict(self._row_cache[vacancy_id]) | {
                "description": new_text
            }
        self._rerate(vacancy_id)

    def action_edit_reason(self) -> None:
        if self.view == "duplicates" or self._selected_id is None:
            return
        self.edit_reason_for_id(self._selected_id)

    def edit_reason_for_id(self, vacancy_id: str) -> None:
        # Read from the store, not the row cache -- the cache may hold a raw
        # sqlite3.Row (no .get()) or a plain dict depending on whether it's
        # been through a prior in-place update, and the store is the
        # authoritative current value either way.
        current_row = self.store.conn.execute(
            "SELECT reason FROM vacancy_status WHERE vacancy_id = ?", (vacancy_id,)
        ).fetchone()
        current = (current_row["reason"] if current_row else None) or ""
        new_text = self._prompt_editor(current)
        if new_text is None or new_text == current:
            return
        try:
            self.store.set_reason(vacancy_id, new_text)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        if vacancy_id in self._row_cache:
            self._row_cache[vacancy_id] = dict(self._row_cache[vacancy_id]) | {"reason": new_text}

    def _prompt_editor(self, current: str) -> str:
        """Isolated from edit_description_for_id so tests can monkeypatch
        just this method -- suspending the app to run a real subprocess
        editor is not something to exercise under `App.run_test()`'s
        headless driver, which is not guaranteed to support suspending."""
        with self.suspend():
            return open_in_editor(current)

    @work(thread=True)
    def _rerate(self, vacancy_id: str) -> None:
        llm = self._llm_override or build_llm(self._config)
        row = self.store.get_vacancy(vacancy_id)
        try:
            outcome = rate_and_save(self.store, self.candidate, llm, row, self.candidate.hash())
        except LLMError as exc:
            hint = llm_hint(self._config)
            message = f"{exc}\n{hint}" if hint else str(exc)
            self.call_from_thread(self.notify, message, severity="error")
            return
        if not outcome.ok:
            self.call_from_thread(self.notify, f"re-rate failed: {outcome.error}", severity="error")
            return
        self.call_from_thread(self._apply_rerate, vacancy_id, outcome.final_score)

    def _apply_rerate(self, vacancy_id: str, score: int) -> None:
        if vacancy_id in self._row_cache:
            self._row_cache[vacancy_id] = dict(self._row_cache[vacancy_id]) | {"score": score}
        if vacancy_id in self._loaded_ids and self.view != "duplicates":
            self.query_one("#table", DataTable).update_cell(vacancy_id, self._columns[0], str(score))

    def action_unmark_duplicate(self) -> None:
        if self.view != "duplicates" or self._selected_id is None:
            return
        self.store.unmark_duplicate(self._selected_id)
        if self._selected_id in self._loaded_ids:
            self.query_one("#table", DataTable).remove_row(self._selected_id)
            self._loaded_ids.discard(self._selected_id)

    def mark_status_for_id(self, vacancy_id: str, status: str) -> None:
        self.store.set_status(vacancy_id, status)
        if vacancy_id in self._row_cache:
            self._row_cache[vacancy_id] = dict(self._row_cache[vacancy_id]) | {"status": status}

        still_in_view = (
            (self.view == "active" and status not in ("dismissed", "closed"))
            or (self.view not in ("active", "duplicates") and self.view == status)
        )
        table = self.query_one("#table", DataTable)
        if still_in_view:
            if vacancy_id in self._loaded_ids:
                table.update_cell(vacancy_id, self._columns[2], status)
        else:
            if vacancy_id in self._loaded_ids:
                table.remove_row(vacancy_id)
                self._loaded_ids.discard(vacancy_id)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        self.query_one("#table", DataTable).focus()

    def action_cancel_search(self) -> None:
        if self.focused is not self.query_one("#search", Input):
            return
        self.query_one("#search", Input).clear()
        self.query_one("#table", DataTable).focus()

    def _get_last_rated_at(self) -> str:
        lr_at = self.store.get_profile_last_rated_at(self.candidate.hash())
        if lr_at is None:
            return "Not rated yet"
        lr_at = datetime.datetime.fromisoformat(lr_at["last_rated_at"])
        if lr_at.date() == datetime.date.today():
            return f"Rated today at {lr_at.time().strftime('%H:%M')}"

        return f"Rated at {lr_at.date().strftime('%Y-%m-%d %H:%M')} "
