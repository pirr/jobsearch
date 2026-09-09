"""Shared $EDITOR flow for pasting a full posting/description by hand.

Used by both the CLI (`update-description`, run interactively) and the TUI
(`e` in the vacancy list) -- the only difference between the two callers is
what happens to the terminal around the subprocess call, which each caller
owns itself (the TUI wraps this in `app.suspend()`; the CLI just calls it
directly, since it already owns the terminal).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def open_in_editor(current: str, *, run=subprocess.run) -> str:
    """Write `current` to a tempfile, run $EDITOR on it, return the result.

    `run` is injectable so tests exercise this without shelling out to a
    real editor.
    """
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(current)
        path = Path(f.name)
    try:
        run([editor, str(path)], check=True)
        return path.read_text(encoding="utf-8").strip()
    finally:
        path.unlink(missing_ok=True)
