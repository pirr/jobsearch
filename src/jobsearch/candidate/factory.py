"""Extension -> reader registry (task.md:19: factory + dependency injection).

Readers register themselves here; nothing else in the system knows which file
formats exist. Adding .odt later means adding one adapter and one line.
"""

from __future__ import annotations

from pathlib import Path

from .readers.base import DocumentReader, ReaderError
from .readers.docx import DocxReader
from .readers.pdf import PdfReader
from .readers.text import TextReader

DEFAULT_READERS: tuple[DocumentReader, ...] = (PdfReader(), DocxReader(), TextReader())


class ReaderFactory:
    def __init__(self, readers: tuple[DocumentReader, ...] = DEFAULT_READERS):
        self._by_ext: dict[str, DocumentReader] = {}
        for reader in readers:
            for ext in reader.extensions:
                self._by_ext[ext.lower()] = reader

    @property
    def supported(self) -> list[str]:
        return sorted(self._by_ext)

    def for_path(self, path: Path) -> DocumentReader:
        reader = self._by_ext.get(path.suffix.lower())
        if reader is None:
            raise ReaderError(
                f"no reader for {path.suffix or '(no extension)'}; "
                f"supported: {', '.join(self.supported)}"
            )
        return reader

    def read(self, path: Path) -> str:
        path = Path(path)
        if not path.exists():
            raise ReaderError(f"{path} does not exist")
        return self.for_path(path).read(path)

    def read_all(self, paths) -> str:
        """Concatenate several documents -- CV plus a cover letter, say."""
        chunks = []
        for p in paths:
            p = Path(p)
            chunks.append(f"--- {p.name} ---\n{self.read(p)}")
        return "\n\n".join(chunks)
