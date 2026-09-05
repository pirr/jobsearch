from __future__ import annotations

from pathlib import Path

from .base import DocumentReader, ReaderError


class TextReader(DocumentReader):
    extensions = (".txt", ".md", ".markdown", ".rst")

    def read(self, path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise ReaderError(f"could not read {path}: {exc}") from exc
        raise ReaderError(f"could not decode {path} as text")
