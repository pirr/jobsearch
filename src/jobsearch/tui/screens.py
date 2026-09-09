"""Secondary screens for the TUI: the no-profile/no-ratings guard, and the
full-vacancy detail view."""

from __future__ import annotations

import json

from rich.markup import escape as rich_escape
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


class DetailScreen(Screen):
    """Full detail for one vacancy -- the TUI equivalent of `_panel()` in
    cli/main.py's `show --full`."""

    BINDINGS = [
        ("escape", "close", "Back"),
        ("q", "close", "Back"),
        ("s", "mark_status('seen')", "Seen"),
        ("d", "mark_status('dismissed')", "Dismissed"),
        ("a", "mark_status('applied')", "Applied"),
        ("m", "mark_duplicate", "Duplicate"),
    ]

    def __init__(self, row) -> None:
        super().__init__()
        self.row = row

    def compose(self) -> ComposeResult:
        yield Static(self._body_text(), id="detail-body")
        yield Footer()

    def _body_text(self) -> str:
        row = self.row
        matched = json.loads(row["matched_skills"] or "[]")
        missing = json.loads(row["missing_skills"] or "[]")
        blockers = json.loads(row["hard_blockers"] or "[]")
        penalty_notes = json.loads(row["evidence"] or "{}").get("penalty_notes") or []

        lines = [
            f"[bold]{row['score']}[/] · {row['title']}",
            f"[bold]{row['company']}[/] · {row['location_raw'] or 'location unspecified'}",
            f'[link="{row["url"]}"]{row["url"]}[/link]',
            f"status: {row['status']}",
            "",
        ]
        if matched:
            lines.append("[green]fits:[/]")
            for m in matched[:6]:
                lines.append(f"  + {m.get('skill', '')}")
                lines.append(f"    [dim]CV: “{m.get('evidence_cv', '')[:88]}”[/]")
                lines.append(f"    [dim]job: “{m.get('evidence_vacancy', '')[:88]}”[/]")
        if missing:
            lines.append(f"[yellow]missing:[/] {', '.join(missing[:8])}")
        if blockers:
            lines.append(f"[red]blockers:[/] {'; '.join(blockers[:4])}")
        if penalty_notes:
            lines.append("[dim]soft penalties:[/]")
            for note in penalty_notes:
                lines.append(f"  [dim]{rich_escape(note)}[/]")
        return "\n".join(lines)

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_mark_status(self, status: str) -> None:
        self.app.mark_status_for_id(self.row["id"], status)
        self.app.pop_screen()

    def action_mark_duplicate(self) -> None:
        self.app.pop_screen()
        self.app.open_duplicate_picker(self.row["id"], self.row["title"], self.row["company"])
