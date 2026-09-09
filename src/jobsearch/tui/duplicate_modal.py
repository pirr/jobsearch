"""Modal for the manual mark-duplicate escape hatch: dedupe.py's hash-based
identity misses near-duplicates (typo'd company, differently formatted
location, a snippet that differs across two sources' APIs for the same
role), so a human picks the pair here."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class DuplicatePickerModal(ModalScreen[str | None]):
    """Search all vacancies and pick one as the canonical for `vacancy_id`.

    Dismisses with the chosen canonical_id, or None if cancelled -- the
    caller (JobsearchApp) does the actual `store.mark_duplicate` call and
    the resulting local table update; this modal only picks the target.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, store, vacancy_id: str, title: str, company: str) -> None:
        super().__init__()
        self.store = store
        self.vacancy_id = vacancy_id
        self._title, self._company = title, company
        self._results: list = []
        self._chosen = None

    def compose(self) -> ComposeResult:
        with Vertical(id="duplicate-modal"):
            yield Label(f"Mark “{self._title}” ({self._company}) as a duplicate of…")
            yield Input(placeholder="search title / company / id", id="dup-search")
            yield ListView(id="dup-results")
            yield Static("", id="dup-confirm")
            with Horizontal(id="dup-buttons"):
                yield Button("Confirm", id="dup-confirm-btn", disabled=True)
                yield Button("Cancel", id="dup-cancel-btn")

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip()
        results_view = self.query_one("#dup-results", ListView)
        results_view.clear()
        self._results = []
        self._set_chosen(None)
        if len(query) < 2:
            return
        for row in self.store.search_vacancies(query, limit=20):
            if row["id"] == self.vacancy_id:
                continue
            self._results.append(row)
            results_view.append(
                ListItem(Label(f"{row['title']} — {row['company']} [{row['id'][:8]}]"))
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#dup-results", ListView).index
        if index is None or index >= len(self._results):
            return
        self._set_chosen(self._results[index])

    def _set_chosen(self, row) -> None:
        self._chosen = row
        confirm = self.query_one("#dup-confirm", Static)
        button = self.query_one("#dup-confirm-btn", Button)
        if row is None:
            confirm.update("")
            button.disabled = True
        else:
            confirm.update(f"Hide as a duplicate of “{row['title']}” ({row['company']})?")
            button.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dup-confirm-btn" and self._chosen is not None:
            self.dismiss(self._chosen["id"])
        elif event.button.id == "dup-cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
