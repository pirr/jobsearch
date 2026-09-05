"""The document reader port (task.md:19)."""

from __future__ import annotations

import abc
from pathlib import Path


class ReaderError(RuntimeError):
    pass


class DocumentReader(abc.ABC):
    #: Lower-case extensions this reader claims, including the dot.
    extensions: tuple[str, ...] = ()

    @abc.abstractmethod
    def read(self, path: Path) -> str:
        """Return the document's plain text."""
