"""The normalized vacancy every source adapter must produce.

Sources disagree about almost everything -- description encoding, location
shape, timestamp format -- so this is the only vacancy representation the rest
of the system is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Vacancy:
    source: str
    source_id: str
    url: str
    company: str
    title: str
    description: str          # plain text, HTML already stripped
    location_raw: str = ""
    location_city: str | None = None
    location_country: str | None = None
    is_remote: bool | None = None
    employment_type: str | None = None
    posted_at: datetime | None = None
    raw: dict = field(default_factory=dict)

    id: str = ""              # dedupe key, filled by dedupe.vacancy_id()

    def searchable_text(self) -> str:
        """The text retrieval and rating actually read."""
        parts = [self.title, self.company, self.location_raw, self.description]
        return "\n".join(p for p in parts if p)
