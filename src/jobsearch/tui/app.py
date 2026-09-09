"""The interactive terminal UI's main app: filter bar + vacancy list.

Like cli/main.py, this only ever reads/writes through Store -- no parallel
data-access path, no business logic duplicated into the UI layer.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Input, Select, Static, Switch

from ..candidate.profile import CandidateProfile
from ..config import load_config
from ..vacancies.store import Store
from .screens import NoProfileScreen

_VIEWS = ["active", "dismissed", "closed", "duplicates"]
_SORT_OPTIONS = [
    ("Score", "score"), ("Date posted", "posted_at"),
    ("Company", "company"), ("Title", "title"),
]


class JobsearchApp(App):
    """Browse, filter and act on rated vacancies without a display limit."""

    CSS = """
    #filterbar { height: 3; }
    #filterbar Input { width: 1fr; }
    #filterbar Select { width: 22; }
    """
    TITLE = "jobsearch"

    BINDINGS = [
        Binding("s", "mark_status('seen')", "Seen"),
        Binding("d", "mark_status('dismissed')", "Dismissed"),
        Binding("a", "mark_status('applied')", "Applied"),
        Binding("q", "quit", "Quit"),
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
        with Horizontal(id="filterbar"):
            yield Input(placeholder="search title/company/location", id="search")
            yield Select(_SORT_OPTIONS, value="score", id="sort", allow_blank=False)
            yield Switch(id="remote-only")
            yield Static("remote only", id="remote-only-label")
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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._selected_id = str(event.row_key.value) if event.row_key.value else None

    def action_mark_status(self, status: str) -> None:
        if self.view == "duplicates" or self._selected_id is None:
            return
        self.mark_status_for_id(self._selected_id, status)

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
