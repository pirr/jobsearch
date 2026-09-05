"""CV text -> CandidateProfile, via the LLM port."""

from __future__ import annotations

from ..matching.llm.base import LLMClient, LLMError
from .profile import CandidateProfile, Constraint

SYSTEM = """You extract structured candidate profiles from CV text.
You return only JSON matching the requested schema. You never invent facts."""

PROMPT = """Extract a candidate profile from the CV below.

Return JSON with exactly these keys:
- "name": string
- "headline": string, a one-line professional summary
- "summary": string, 2-3 sentences on experience and focus
- "years_experience": number or null
- "skills": array of strings (technologies, tools, languages, frameworks)
- "titles": array of strings (job titles held)
- "industries": array of strings
- "languages": array of strings (spoken languages)
- "constraints": array of objects, each with:
    - "kind": one of "remote","location","seniority","language","employment_type","visa"
    - "value": string
    - "stated": boolean
    - "evidence": string

CRITICAL RULE for constraints. Set "stated": true ONLY when the CV explicitly
declares the preference or requirement, and put the exact quoted sentence in
"evidence". Examples of stated constraints: "Looking for remote-only roles",
"Requires visa sponsorship", "Available for full-time positions in Berlin".

If you are merely inferring from context -- the candidate's current city, their
seniority from years worked, languages implied by employers -- set
"stated": false and leave "evidence" empty. Do not guess about relocation
willingness, salary, or visa status unless the CV says so.

Never write a constraint whose evidence is not a verbatim quote from the CV.

CV:
---
{cv_text}
---"""

MAX_CV_CHARS = 24000


def extract_profile(
    llm: LLMClient, cv_text: str, source_files: list[str] | None = None
) -> CandidateProfile:
    if not cv_text.strip():
        raise ValueError("no text extracted from the given documents")

    data = llm.complete_json(SYSTEM, PROMPT.format(cv_text=cv_text[:MAX_CV_CHARS]))
    if not isinstance(data, dict):
        raise LLMError(f"expected a JSON object, got {type(data).__name__}")

    constraints = []
    for item in data.get("constraints") or []:
        if not isinstance(item, dict):
            continue
        try:
            constraints.append(Constraint(**item))
        except Exception:
            continue  # a malformed constraint is dropped, never guessed at

    profile = CandidateProfile(
        name=str(data.get("name") or ""),
        headline=str(data.get("headline") or ""),
        summary=str(data.get("summary") or ""),
        years_experience=_number(data.get("years_experience")),
        skills=_strings(data.get("skills")),
        titles=_strings(data.get("titles")),
        industries=_strings(data.get("industries")),
        languages=_strings(data.get("languages")),
        constraints=constraints,
        source_files=source_files or [],
    )
    return _verify_evidence(profile, cv_text)


def _verify_evidence(profile: CandidateProfile, cv_text: str) -> CandidateProfile:
    """Demote any constraint whose quoted evidence is not actually in the CV.

    Models fabricate plausible-sounding quotes. An unverifiable quote must not
    be allowed to become a hard filter, so it is demoted to a soft signal
    rather than trusted or silently dropped.
    """
    haystack = " ".join(cv_text.lower().split())
    for constraint in profile.constraints:
        if not constraint.stated:
            continue
        quote = " ".join(constraint.evidence.lower().split())
        if not quote or quote not in haystack:
            constraint.stated = False
            constraint.evidence = ""
    return profile


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    seen, out = set(), []
    for item in value:
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
