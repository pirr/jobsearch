from __future__ import annotations

from pathlib import Path

from .base import DocumentReader, ReaderError


class PdfReader(DocumentReader):
    extensions = (".pdf",)

    def read(self, path: Path) -> str:
        try:
            from pypdf import PdfReader as _PdfReader
        except ImportError as exc:
            raise ReaderError("pypdf is required to read PDF files") from exc
        try:
            reader = _PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise ReaderError(f"could not read PDF {path}: {exc}") from exc
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text.strip():
            # A scanned CV extracts to nothing. Saying so beats handing the LLM
            # an empty string and returning a confidently empty profile.
            raise ReaderError(
                f"{path} produced no extractable text -- it may be a scanned image, "
                "which needs OCR (out of scope for the MVP)"
            )
        return text
