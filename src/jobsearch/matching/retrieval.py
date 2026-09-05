"""Cheap shortlisting before any LLM call.

Rating every vacancy is what makes a run unaffordable: the corpus is roughly
4,600 postings, and a per-vacancy LLM call across all of them costs real money
on cloud and hours on a local model. Retrieval narrows that to a shortlist the
LLM can rerank properly.

Two implementations behind one port so the golden set decides between them.
"""

from __future__ import annotations

import abc
import math
import re
from collections import Counter
from collections.abc import Sequence

from .llm.base import LLMError

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Frequent in every posting and so carry no signal about fit.
_STOPWORDS = frozenset("""
a an the and or of to in for with on at by from as is are be will you your we our us
their its this that these those it they he she them who whom which what when where
how why all any both each few more most other some such no nor not only own same so
than too very can just should now team role job work working position opportunity
company please apply candidate candidates experience years about have has had do does
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


class Retriever(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score per document, same order as input."""

    def top_n(self, query: str, documents: Sequence[str], n: int) -> list[tuple[int, float]]:
        scores = self.rank(query, documents)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:n]]


class BM25Retriever(Retriever):
    """Okapi BM25. No dependencies, fully explainable, blind to synonyms."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        docs = [tokenize(d) for d in documents]
        lengths = [len(d) for d in docs]
        avg_len = (sum(lengths) / len(lengths)) or 1.0
        n_docs = len(docs)

        doc_freq: Counter[str] = Counter()
        for tokens in docs:
            doc_freq.update(set(tokens))

        q_tokens = set(tokenize(query))
        term_freqs = [Counter(d) for d in docs]

        scores = []
        for idx, tf in enumerate(term_freqs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * lengths[idx] / avg_len)
            for term in q_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (freq * (self.k1 + 1)) / (freq + norm)
            scores.append(score)
        return scores


class EmbeddingRetriever(Retriever):
    """Cosine similarity over embeddings from the LLM port.

    Catches what BM25 cannot -- "k8s" against "Kubernetes", "ML" against
    "machine learning" -- at the cost of one embedding pass over the corpus.
    """

    name = "embeddings"

    def __init__(self, llm, batch_size: int = 64, max_chars: int = 1500):
        # 1500 chars is a conservative default, not a measured token count --
        # real job postings were observed tokenizing as densely as ~0.8
        # tokens/char (technical jargon, bullet-heavy formatting), and a
        # locally-served embedding model's context window can be as small as
        # 2048 tokens. There is no way to know a given server's actual limit
        # from the client side, so rank() also degrades per-item on a context
        # error rather than trusting this number to always be safe.
        self.llm = llm
        self.batch_size = batch_size
        self.max_chars = max_chars

    def rank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        q_vec = self._embed_one(query)
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            try:
                vectors = self.llm.embed([d[: self.max_chars] for d in batch])
            except LLMError:
                # One oversized document can fail an entire batch. Falling
                # back to one-at-a-time isolates it instead of losing the
                # other 63 documents' scores to a single bad one.
                vectors = [self._embed_one(d) for d in batch]
            scores.extend(_cosine(q_vec, vec) for vec in vectors)
        return scores

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text, shrinking it if the server rejects the length.

        A silent 0.0 score for a document that never embeds is preferable to
        aborting the whole shortlist over one pathologically dense posting.
        """
        size = self.max_chars
        for _ in range(4):
            try:
                return self.llm.embed([text[:size]])[0]
            except LLMError:
                size //= 2
                if size < 50:
                    return []
        return []


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_retriever(name: str, llm=None) -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    if name == "embeddings":
        if llm is None:
            raise ValueError("the embeddings retriever needs an LLM client")
        return EmbeddingRetriever(llm)
    raise ValueError(f"unknown retriever {name!r}; use bm25 or embeddings")
