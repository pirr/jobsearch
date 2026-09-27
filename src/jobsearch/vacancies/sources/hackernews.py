"""Hacker News "Ask HN: Who is hiring?" -- a breadth source read through the
public Algolia HN API (hn.algolia.com), verified live 2026-09-27.

Each month's thread is posted by the `whoishiring` account; every top-level
comment is one company's post, and replies are discussion, never postings.
There is no structured schema: posts conventionally open with a header like
`Company | Role | Location | Full-time | Salary`, which is parsed
best-effort here. The full post goes into `description` either way, so the
LLM rater still sees everything the header parse misses.

Only the newest hiring thread is read. Last month's posts already ingested
simply stay in the database, as for every other breadth source.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import httpx

from ..models import Vacancy
from ..normalize import normalize_location, normalize_timestamp, strip_html
from .base import SourceError, VacancySource, get_json

API = "https://hn.algolia.com/api/v1"
THREAD_TITLE = "Ask HN: Who is hiring?"

# The header is everything before the first paragraph break. `|` is the
# convention; a few posters use `·` instead. A post with neither is skipped:
# in a recorded thread those were ~6% of comments and mostly meta-chatter
# ("Please normalize 4DWW", thread-navigation tools) rather than postings.
_SEPARATOR_RE = re.compile(r"\s*[|·]\s*")
_URL_RE = re.compile(r"https?://\S+")
_ROLE_RE = re.compile(
    r"engineer|developer|scientist|researcher|designer|manager|architect|analyst"
    r"|devops|\bsre\b|\bswe\b|\broles?\b|positions?|openings?|\bcto\b|head of"
    r"|founding|operator|programmer|consultant|specialist|\blead\b|\bintern\b",
    re.IGNORECASE,
)
_PLACE_RE = re.compile(
    r"remote|on-?site|on site|hybrid|in-person|in person|office|worldwide|anywhere"
    r"|global|emea|europe|\b(?:us|usa|uk|eu)\b|timezone|\butc\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"[$€£]|\d+\s*k\b", re.IGNORECASE)
_EMPLOYMENT_RE = re.compile(
    r"full[- ]?time|part[- ]?time|contract|freelance|internship", re.IGNORECASE
)
# "REMOTE WILL BE IGNORED" mentions remote and means the opposite.
_NOT_REMOTE_RE = re.compile(
    r"no remote|not remote|remote will be ignored|remote not (?:possible|available)",
    re.IGNORECASE,
)
_ONSITE_RE = re.compile(r"on-?site|on site|in-person|in person|hybrid", re.IGNORECASE)


class HackerNewsSource(VacancySource):
    name = "hackernews"

    async def fetch(self, client: httpx.AsyncClient) -> Sequence[Vacancy]:
        search = await get_json(
            client,
            f"{API}/search_by_date",
            params={"tags": "story,author_whoishiring", "hitsPerPage": 5},
        )
        # The same account also posts "Who wants to be hired?" (candidates,
        # not jobs) and "Freelancer? Seeking freelancer?" the same day.
        threads = [
            h for h in search.get("hits") or []
            if (h.get("title") or "").startswith(THREAD_TITLE)
        ]
        if not threads:
            raise SourceError(f"no {THREAD_TITLE!r} thread in the latest whoishiring posts")
        thread = await get_json(client, f"{API}/items/{threads[0]['objectID']}")
        out = [self._to_vacancy(c) for c in thread.get("children") or []]
        return self.finalize([v for v in out if v is not None])

    def _to_vacancy(self, comment: dict) -> Vacancy | None:
        body = comment.get("text")
        if not body:  # deleted or flagged comments arrive with text=None
            return None
        header = strip_html(re.split(r"<p>", body, maxsplit=1)[0])
        segments = [
            s for s in (_SEPARATOR_RE.split(header.replace("\n", " "))) if s.strip()
        ]
        if len(segments) < 2:
            return None

        company = re.sub(r"\(\s*\)", "", _URL_RE.sub("", segments[0]))
        company = re.sub(r"\s+", " ", company).strip(" *-–")
        rest = [s.strip(" *") for s in segments[1:] if _URL_RE.sub("", s).strip()]
        title = next((s for s in rest if _ROLE_RE.search(s)), "")
        places = [
            s for s in rest
            if s != title and not _MONEY_RE.search(s)
            and (_PLACE_RE.search(s) or ("," in s and not re.search(r"\d", s)))
        ]
        employment = next(
            (s for s in rest if _EMPLOYMENT_RE.search(s) and len(s) < 40), None
        )
        raw_loc, _, country, _ = normalize_location(" | ".join(places))

        is_remote: bool | None = None
        if _NOT_REMOTE_RE.search(header):
            is_remote = False
        elif re.search(r"remote", raw_loc, re.IGNORECASE):
            is_remote = True
        elif _ONSITE_RE.search(raw_loc):
            is_remote = False

        comment_id = str(comment.get("id", ""))
        return Vacancy(
            source=self.name,
            source_id=comment_id,
            url=f"https://news.ycombinator.com/item?id={comment_id}",
            company=company[:80],
            # No role named in the header ("Smarkets | Full Time | London"):
            # the post body lists roles, so the title says so instead of
            # guessing -- never empty, since every view keys off it.
            title=(title or "Open roles (see post)")[:150],
            description=strip_html(body),
            location_raw=raw_loc,
            # The header's place segment is free text ("Remote (Europe)",
            # "Chicago, IL / Remote") -- a city guess from it would be wrong
            # more often than right, so only the country hint is kept.
            location_city=None,
            location_country=country,
            is_remote=is_remote,
            employment_type=employment,
            posted_at=normalize_timestamp(comment.get("created_at")),
            raw={k: v for k, v in comment.items() if k != "children"},
        )
