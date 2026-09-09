"""LLM reranking of the shortlist, with grounded explanations.

A free-text "why it fits" invents requirements the posting never listed and
skills the CV never claimed. So the model must quote a span from BOTH documents
for every skill it calls a match, and any quote that is not actually present is
discarded before the score is stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from ..candidate.profile import CandidateProfile
from . import eligibility
from .llm.base import LLMClient, LLMError

SYSTEM = """You rate how well a candidate fits a job vacancy.
You return only JSON. Every claim you make must be supported by a verbatim
quote from the source documents. You never invent requirements or skills."""

PROMPT = """Rate this candidate against this vacancy.

Return JSON with exactly these keys:
- "score": integer 0-100, how well the candidate fits
- "hard_blockers": array of strings -- requirements the candidate clearly cannot
  meet: a missing mandatory qualification, wrong field entirely, OR the
  vacancy being remote-in-name-only -- advertised as remote but actually
  restricted to a region the candidate is not eligible to work in (see the
  geo/visa signal below), with no visa sponsorship offered. Do NOT put an
  ordinary on-site/hybrid role simply being in a different city or country
  than the candidate here -- that is already scored separately as a location
  preference, not a hard blocker. Empty if none.
- "matched_skills": array of objects, each with:
    - "skill": string
    - "evidence_cv": verbatim quote from the CANDIDATE PROFILE proving they have it
    - "evidence_vacancy": verbatim quote from the VACANCY proving it is wanted
- "missing_skills": array of strings the vacancy asks for and the candidate lacks
- "summary_fit": 1-2 sentences on why this fits
- "summary_gap": 1-2 sentences on why it might not fit

Rules:
- Quotes must be copied EXACTLY from the text below. Do not paraphrase.
- If you cannot find a real quote for a skill, leave that skill out.
- A remote-in-name-only geographic/visa restriction (see hard_blockers above)
  is a hard blocker even when every technical skill matches -- do not let a
  strong technical fit hide it. An ordinary on-site/hybrid location
  difference is NOT a hard blocker on its own, no matter how far it is from
  the candidate's location.
- Score bands: 80-100 strong fit, 60-79 worth applying, 40-59 marginal,
  0-39 poor fit.

CANDIDATE PROFILE:
---
{profile}
---
CANDIDATE CONSTRAINTS (for context only -- "inferred" ones are a guess, not
stated in the CV; do not treat an ordinary on-site/hybrid location difference
against these as a hard blocker, per the hard_blockers rule above):
{constraints}

VACANCY:
Title: {title}
Company: {company}
Location: {location}
Geo/visa signal (heuristic keyword match over this vacancy's own text --
verify against the text below rather than trusting it blindly): {geo_signal}
---
{description}
---"""

MAX_DESCRIPTION_CHARS = 12000


class MatchedSkill(BaseModel):
    skill: str
    evidence_cv: str = ""
    evidence_vacancy: str = ""


class Rating(BaseModel):
    score: int = Field(ge=0, le=100)
    hard_blockers: list[str] = Field(default_factory=list)
    matched_skills: list[MatchedSkill] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    summary_fit: str = ""
    summary_gap: str = ""


@dataclass
class RatingOutcome:
    vacancy_id: str
    rating: Rating | None = None
    error: str | None = None
    dropped_claims: list[str] = field(default_factory=list)
    penalty: int = 0
    penalty_notes: list[str] = field(default_factory=list)
    # Populated by pipeline.rate_and_save(), not by rate_one() itself: the
    # clamped, hard-blocker-capped score actually persisted, so a caller that
    # needs it (a single-vacancy re-rate command, say) has one place to read
    # it instead of re-deriving the same clamp math a second time.
    final_score: int | None = None

    @property
    def ok(self) -> bool:
        return self.rating is not None


def _normalize(text: str) -> str:
    # Models reflow whitespace and swap quote characters when copying spans;
    # neither means the quote is fabricated.
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    return " ".join(text.split())


def _grounded(quote: str, haystack: str) -> bool:
    q = _normalize(quote)
    if len(q) < 8:
        return False  # too short to prove anything
    return q in haystack


def verify_grounding(rating: Rating, profile_text: str, vacancy_text: str) -> list[str]:
    """Drop matched skills whose quotes are not in the sources. Returns what went."""
    cv_hay, vac_hay = _normalize(profile_text), _normalize(vacancy_text)
    kept, dropped = [], []
    for skill in rating.matched_skills:
        if _grounded(skill.evidence_cv, cv_hay) and _grounded(skill.evidence_vacancy, vac_hay):
            kept.append(skill)
        else:
            dropped.append(skill.skill)
    rating.matched_skills = kept
    return dropped


def _constraints_text(profile: CandidateProfile) -> str:
    if not profile.constraints:
        return "none recorded"
    lines = [
        f"- {c.kind}: {c.value} ({'stated in CV' if c.stated else 'inferred, not stated in CV'})"
        for c in profile.constraints
    ]
    return "\n".join(lines)


def _geo_note(elig: eligibility.GeoEligibility) -> str:
    note = eligibility.describe(elig)
    if elig.region_restricted is None:
        # Reinforce the prompt's own rule at the exact point the model reads
        # the signal, for the common case (a plain on-site/hybrid posting)
        # where it would otherwise be tempted to hard-block on location alone.
        note += " -- do not treat this vacancy's plain location as a hard blocker"
    return note


def rate_one(
    llm: LLMClient,
    profile: CandidateProfile,
    row,
    *,
    retries: int = 1,
) -> RatingOutcome:
    profile_text = profile.searchable_text()
    description = (row["description"] or "")[:MAX_DESCRIPTION_CHARS]
    prompt = PROMPT.format(
        profile=profile_text,
        constraints=_constraints_text(profile),
        title=row["title"],
        company=row["company"],
        location=row["location_raw"] or "unspecified",
        geo_signal=_geo_note(eligibility.classify(row)),
        description=description,
    )

    last_error = None
    for _ in range(retries + 1):
        try:
            data = llm.complete_json(SYSTEM, prompt)
            rating = Rating.model_validate(data)
        except (LLMError, ValidationError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        dropped = verify_grounding(rating, profile_text, description)
        return RatingOutcome(row["id"], rating=rating, dropped_claims=dropped)

    # Recorded as failed rather than silently dropped, so `show` can say how
    # many vacancies never got a verdict.
    return RatingOutcome(row["id"], error=last_error or "unknown rating failure")


def verdict_for(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "worth applying"
    if score >= 40:
        return "marginal"
    return "poor"
