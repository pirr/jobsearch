import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path):
    from jobsearch.vacancies.store import Store

    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def profile():
    from jobsearch.candidate.profile import CandidateProfile, Constraint

    return CandidateProfile(
        headline="Senior Python Backend Engineer",
        summary="Eight years of Python backend work.",
        skills=["Python", "Django", "Kubernetes", "PostgreSQL"],
        titles=["Senior Backend Engineer"],
        constraints=[
            Constraint(kind="remote", value="remote only", stated=True,
                       evidence="I am looking for remote-only roles"),
            Constraint(kind="location", value="Berlin", stated=False),
        ],
    )


def make_vacancy(**kw):
    from jobsearch.vacancies.dedupe import vacancy_id
    from jobsearch.vacancies.models import Vacancy

    defaults = dict(source="test", source_id="1", url="https://example.com/1",
                    company="Acme", title="Backend Engineer",
                    description="We need Python and Kubernetes experience.")
    defaults.update(kw)
    v = Vacancy(**defaults)
    v.id = vacancy_id(v.company, v.title, v.location_raw, v.description)
    return v


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._payload


class FakeClient:
    """Stands in for httpx.AsyncClient so adapters run on recorded payloads."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests = []

    async def get(self, url, params=None):
        self.requests.append((url, params))
        payload = self.payload
        if callable(payload):
            payload = payload(len(self.requests))
        return FakeResponse(payload, self.status_code)
