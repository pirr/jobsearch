"""The candidate profile, and the hard/soft distinction that protects it.

Constraints are inferred from the CV, so some of them are guesses about things
the document never actually says -- relocation willingness, salary floor, visa
status. A wrong guess applied as a filter deletes good matches invisibly, so
only constraints the CV states outright are allowed to exclude anything.
Everything else is a scoring penalty the candidate can see and overrule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

CONSTRAINT_KINDS = (
    "remote", "location", "seniority", "language", "employment_type", "visa",
)


class Constraint(BaseModel):
    kind: str
    value: str
    stated: bool = False   # True only when the CV says it outright
    evidence: str = ""     # the quoted span that justifies `stated`

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def is_hard(self) -> bool:
        """A constraint filters only if the CV stated it and quoted proof exists."""
        return self.stated and bool(self.evidence.strip())


class CandidateProfile(BaseModel):
    name: str = ""
    headline: str = ""
    summary: str = ""
    years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    #: Kept so `match` can show what it read, and so re-extraction is auditable.
    source_files: list[str] = Field(default_factory=list)

    @property
    def hard_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.is_hard]

    @property
    def soft_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if not c.is_hard]

    def searchable_text(self) -> str:
        """What the retriever compares against vacancy text."""
        parts = [
            self.headline, self.summary,
            " ".join(self.titles), " ".join(self.skills),
            " ".join(self.industries),
        ]
        return "\n".join(p for p in parts if p.strip())

    def hash(self) -> str:
        """Identity of this profile's *content*.

        Ratings are keyed on it, so editing profile.json re-rates while an
        untouched profile never re-pays. Volatile fields are excluded.
        """
        payload = self.model_dump(exclude={"source_files", "name"})
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "CandidateProfile":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
