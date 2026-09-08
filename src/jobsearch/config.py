"""Runtime configuration.

Values come from environment variables so the CLI stays free of flags that
nobody changes between runs. Anything a user does change lives on the command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Every field below uses default_factory rather than a bare default: a bare
# `os.environ.get(...)` default is evaluated once, when this module is first
# imported, so a later `os.environ["X"] = ...` (as tests and `--env` wrappers
# do) would be silently ignored. default_factory re-reads the environment on
# every `load_config()` call instead.
@dataclass(frozen=True)
class Config:
    db_path: Path = field(default_factory=lambda: _env_path("JOBSEARCH_DB", "jobsearch.db"))
    profile_path: Path = field(
        default_factory=lambda: _env_path("JOBSEARCH_PROFILE", "profile.json")
    )
    profile_storage_path: Path = field(
        default_factory=lambda: _env_path("JOBSEARCH_PROFILE_STORAGE", "profile_storage")
    )
    companies_path: Path = field(
        default_factory=lambda: _env_path("JOBSEARCH_COMPANIES", "data/companies.yaml")
    )

    # LLM
    llm_backend: str = field(default_factory=lambda: _env_str("JOBSEARCH_LLM", "ollama"))
    # A local reasoning model given a generous max_tokens can legitimately take
    # several minutes per call -- observed directly: a real rating call was
    # killed client-side at exactly the old 180s default while the server was
    # still working, logged upstream as a 499 (client closed the connection).
    # 600s is a floor, not a guess -- raise it further for slower hardware.
    llm_timeout: float = field(default_factory=lambda: _env_float("JOBSEARCH_LLM_TIMEOUT", 600.0))
    ollama_host: str = field(
        default_factory=lambda: _env_str("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_OLLAMA_MODEL", "llama3.1:8b")
    )
    ollama_embed_model: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    anthropic_model: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_ANTHROPIC_MODEL", "claude-sonnet-5")
    )
    llamacpp_host: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_LLAMACPP_HOST", "http://localhost:8080")
    )
    llamacpp_model: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_LLAMACPP_MODEL", "local")
    )
    llamacpp_embed_model: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_LLAMACPP_EMBED_MODEL", "")
    )
    # Nomic's text embedding models (the documented default above) require a
    # different task-instruction prefix per role for good retrieval quality.
    # Blank these out via env var for a different embedding model that
    # doesn't use this convention.
    embed_query_prefix: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_EMBED_QUERY_PREFIX", "search_query: ")
    )
    embed_document_prefix: str = field(
        default_factory=lambda: _env_str("JOBSEARCH_EMBED_DOCUMENT_PREFIX", "search_document: ")
    )

    # Matching
    retriever: str = field(default_factory=lambda: _env_str("JOBSEARCH_RETRIEVER", "embeddings"))
    shortlist_size: int = field(default_factory=lambda: _env_int("JOBSEARCH_SHORTLIST", 60))
    score_threshold: int = field(default_factory=lambda: _env_int("JOBSEARCH_THRESHOLD", 60))

    # Ingest
    http_timeout: float = 30.0
    http_retries: int = 3
    max_concurrent_sources: int = field(
        default_factory=lambda: _env_int("JOBSEARCH_CONCURRENCY", 6)
    )

    # Adzuna is the only source that needs credentials; empty means "not
    # configured", and build_sources() skips it silently in that case rather
    # than erroring every run for users who never signed up for a key.
    adzuna_app_id: str = field(default_factory=lambda: _env_str("JOBSEARCH_ADZUNA_APP_ID", ""))
    adzuna_app_key: str = field(default_factory=lambda: _env_str("JOBSEARCH_ADZUNA_APP_KEY", ""))


def load_config() -> Config:
    return Config()
