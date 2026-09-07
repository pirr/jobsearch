"""Heuristic geo/visa-eligibility signal for remote vacancies.

The candidate is currently based in Montenegro -- not the US, not the EU. A
vacancy that markets itself as "remote" but actually scopes that to only the
US or only the EU, without offering visa sponsorship, is not reachable in
practice even when it is a great skills match. Structured location fields are
unreliable for this: a real Greenhouse posting can have `location_raw ==
"Remote"` while the *description* is what actually says "within the United
States", so this reads the full vacancy text rather than trusting
`location_country`.

This hardcodes one fact about the current candidate (not in the US or EU)
rather than doing generic geo-matching against the candidate profile --
revisit if that ever changes. It is deliberately a coarse priority signal, not
a filter: `matching/pipeline.py` uses it to order the shortlist, and the LLM
rating stage (`matching/rating.py`) is the backstop that judges eligibility
properly per vacancy, so an imprecise match here only reorders, never hides,
a vacancy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_US_RESTRICTED_RE = re.compile(
    r"remote[\s,\-]*\(?\s*u\.?s\.?a?\.?\)?"
    r"|\bu\.?s\.?a?\.?\s*only\b"
    r"|within the united states"
    r"|must be (?:located|based|residing|authorized to work) in the (?:u\.?s\.?|united states)"
    r"|u\.?s\.?\s*citizens?\s*(?:only|required)"
    r"|must reside in the (?:u\.?s\.?|united states)",
    re.IGNORECASE,
)

_EU_RESTRICTED_RE = re.compile(
    r"remote[\s,\-]*\(?\s*(?:eu|europe|emea)\)?\b"
    r"|within the european union"
    r"|must be (?:located|based|residing) in (?:the eu|europe)"
    r"|eu work permit"
    r"|eu citizens?\s*(?:only|required)",
    re.IGNORECASE,
)

_SPONSOR_DENIED_RE = re.compile(
    r"visa sponsorship is not (?:offered|available|provided)"
    r"|(?:unable|not able) to sponsor"
    r"|does not (?:offer|provide) (?:visa )?sponsorship"
    r"|no (?:visa )?sponsorship\b",
    re.IGNORECASE,
)

_SPONSOR_OFFERED_RE = re.compile(
    r"(?:will|can|does?|happy|able|glad|willing)\s+(?:to\s+)?sponsor"
    r"|sponsors?\s+(?:a\s+)?(?:work\s+)?visas?"
    r"|visa sponsorship (?:is )?available"
    r"|sponsorship (?:is )?available",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeoEligibility:
    region_restricted: str | None = None  # "US", "EU", or None (not restricted)
    visa_sponsorship: bool | None = None  # True offered, False denied, None unmentioned


def classify(row) -> GeoEligibility:
    """Classify a vacancy row's remote-eligibility scope from its text.

    Only meaningful for remote roles: an on-site posting's location mismatch
    is already handled by the existing soft "location" penalty in
    `filters.py`, so an explicitly non-remote row is always unrestricted here.
    """
    if row["is_remote"] is False:
        return GeoEligibility()

    text = f"{row['location_raw'] or ''} {row['description'] or ''}"
    location = (row["location_raw"] or "").strip().lower()

    region: str | None = None
    if location == "europe" or _EU_RESTRICTED_RE.search(text):
        region = "EU"
    if _US_RESTRICTED_RE.search(text):
        # Postings naming both regions are rare; when both patterns fire, the
        # more specific US phrasing wins rather than picking arbitrarily.
        region = "US"

    sponsorship: bool | None = None
    if _SPONSOR_DENIED_RE.search(text):
        sponsorship = False
    elif _SPONSOR_OFFERED_RE.search(text):
        sponsorship = True

    return GeoEligibility(region_restricted=region, visa_sponsorship=sponsorship)


def is_reachable(elig: GeoEligibility) -> bool:
    """True unless the vacancy is region-restricted with no sponsorship offered."""
    if elig.region_restricted is None:
        return True
    return elig.visa_sponsorship is True


def describe(elig: GeoEligibility) -> str:
    """Short human-readable note for the rating prompt."""
    if elig.region_restricted is None:
        return "no geographic restriction detected"
    sponsor = {
        True: "visa sponsorship appears to be offered",
        False: "visa sponsorship appears to be explicitly denied",
        None: "visa sponsorship is not mentioned",
    }[elig.visa_sponsorship]
    return f"appears restricted to {elig.region_restricted}; {sponsor}"
