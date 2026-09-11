from __future__ import annotations

import json

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from jobsearch.vacancies.store import Store


class ProfilePickerModal(ModalScreen[str | None]):

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.profiles = []

    def compose(self) -> ComposeResult:
        self.profiles = self.store.list_profiles()
        profile_names = [json.loads(p["source_files"])[0] for p in self.profiles]
        profile_list_items = tuple(ListItem(Label(pn)) for pn in profile_names)
        with Vertical(id="profile-modal"):
            yield Label("Select a profile to load:")
            yield ListView(
                id="profile_list",
                *profile_list_items
            )

    def on_list_view_selected(self) -> None:
        index = self.query_one("#profile_list", ListView).index
        if index is None or index >= len(self.profiles):
            return

        self.dismiss(self.profiles[index]["id"])

    def action_cancel(self) -> None:
        self.dismiss(None)
