"""Secondary screens for the TUI: the no-profile/no-ratings guard, and
(Task 15) the full-vacancy detail view."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static


class NoProfileScreen(Screen):
    """Shown instead of the list when there is nothing to browse yet --
    mirrors the CLI's `_fail` guard in `show`/`match` rather than opening a
    broken, empty table."""

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(f"\n{self._message}\n\nPress q to quit.", id="no-profile-message")
        yield Footer()

    def action_quit(self) -> None:
        self.app.exit()
