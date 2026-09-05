"""Offline evaluation of retrieval quality.

Without this, every prompt or retriever change is an unmeasurable guess. A
golden file is self-contained -- profile plus labelled vacancies -- so eval runs
with no database and no network, and BM25 scoring is reproducible in CI.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..candidate.profile import CandidateProfile
from .retrieval import Retriever


@dataclass
class GoldenVacancy:
    id: str
    title: str
    company: str
    location_raw: str
    description: str
    relevant: bool

    def text(self) -> str:
        parts = [self.title, self.company, self.location_raw, self.description]
        return "\n".join(p for p in parts if p)


@dataclass
class GoldenSet:
    name: str
    profile: CandidateProfile
    vacancies: list[GoldenVacancy]

    @property
    def relevant_count(self) -> int:
        return sum(1 for v in self.vacancies if v.relevant)

    @classmethod
    def load(cls, path: Path) -> "GoldenSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name") or Path(path).stem,
            profile=CandidateProfile.model_validate(data["profile"]),
            vacancies=[GoldenVacancy(**v) for v in data["vacancies"]],
        )

    @classmethod
    def load_dir(cls, directory: Path) -> list["GoldenSet"]:
        return [cls.load(p) for p in sorted(Path(directory).glob("*.json"))]


def precision_at_k(labels: list[bool], k: int) -> float:
    top = labels[:k]
    return (sum(top) / len(top)) if top else 0.0


def recall_at_k(labels: list[bool], k: int, total_relevant: int) -> float:
    if not total_relevant:
        return 0.0
    return sum(labels[:k]) / total_relevant


def ndcg_at_k(labels: list[bool], k: int) -> float:
    """Binary-gain nDCG: rewards putting relevant results near the top."""
    def dcg(items: list[bool]) -> float:
        return sum(1.0 / math.log2(i + 2) for i, rel in enumerate(items) if rel)

    actual = dcg(labels[:k])
    ideal = dcg(sorted(labels, reverse=True)[:k])
    return (actual / ideal) if ideal else 0.0


@dataclass
class EvalResult:
    golden: str
    retriever: str
    k: int
    precision: float
    recall: float
    ndcg: float
    n_vacancies: int
    n_relevant: int


def evaluate(golden: GoldenSet, retriever: Retriever, k: int = 10) -> EvalResult:
    docs = [v.text() for v in golden.vacancies]
    ranked = retriever.top_n(golden.profile.searchable_text(), docs, len(docs))
    labels = [golden.vacancies[i].relevant for i, _ in ranked]
    return EvalResult(
        golden=golden.name,
        retriever=retriever.name,
        k=k,
        precision=precision_at_k(labels, k),
        recall=recall_at_k(labels, k, golden.relevant_count),
        ndcg=ndcg_at_k(labels, k),
        n_vacancies=len(golden.vacancies),
        n_relevant=golden.relevant_count,
    )
