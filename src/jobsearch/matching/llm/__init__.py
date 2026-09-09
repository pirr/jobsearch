"""LLM backends, selected by config."""

from __future__ import annotations

from ...config import Config
from .base import LLMClient, LLMError, parse_json_response
from .fake import FakeLLM
from .llamacpp import LlamaCppClient
from .ollama import OllamaClient

__all__ = [
    "LLMClient", "LLMError", "FakeLLM", "OllamaClient", "LlamaCppClient",
    "build_llm", "build_embedder", "parse_json_response", "llm_hint",
]


def build_llm(config: Config) -> LLMClient:
    backend = config.llm_backend.lower()
    if backend == "ollama":
        return OllamaClient(
            config.ollama_host, config.ollama_model, config.ollama_embed_model,
            timeout=config.llm_timeout,
            embed_query_prefix=config.embed_query_prefix,
            embed_document_prefix=config.embed_document_prefix,
        )
    if backend == "llamacpp":
        return LlamaCppClient(
            config.llamacpp_host, config.llamacpp_model, config.llamacpp_embed_model or None,
            timeout=config.llm_timeout,
            embed_query_prefix=config.embed_query_prefix,
            embed_document_prefix=config.embed_document_prefix,
        )
    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(config.anthropic_model)
    if backend == "fake":
        # Deterministic stand-in with no model behind it, for exercising the
        # pipeline without a real backend. Never the default.
        return FakeLLM()
    raise LLMError(f"unknown JOBSEARCH_LLM backend {config.llm_backend!r}")


def llm_hint(config: Config) -> str:
    """Backend-appropriate troubleshooting line for an LLMError.

    The three backends fail in different ways for different reasons, so a
    hint naming the wrong one (e.g. always suggesting `ollama serve` while
    JOBSEARCH_LLM=llamacpp) sends someone chasing the wrong server. Shared
    by the CLI and the TUI so both show the same guidance.
    """
    backend = config.llm_backend.lower()
    if backend == "ollama":
        return f"Is Ollama running? Try: ollama serve && ollama pull {config.ollama_model}"
    if backend == "llamacpp":
        return (
            f"Is llama-server running and reachable at {config.llamacpp_host}? "
            "If it's on another machine, check the port and that it's bound to "
            "0.0.0.0 rather than only 127.0.0.1."
        )
    if backend == "anthropic":
        return "Check ANTHROPIC_API_KEY is set and valid."
    return ""


def build_embedder(config: Config) -> LLMClient:
    """Embeddings follow the chat backend when it can serve them.

    Anthropic has no embedding endpoint, so cloud rating pairs with local
    embeddings (Ollama) by default. llama.cpp serves both from the same
    server, so it is used for both when selected.
    """
    backend = config.llm_backend.lower()
    if backend == "llamacpp":
        return LlamaCppClient(
            config.llamacpp_host, config.llamacpp_model, config.llamacpp_embed_model or None,
            timeout=config.llm_timeout,
            embed_query_prefix=config.embed_query_prefix,
            embed_document_prefix=config.embed_document_prefix,
        )
    if backend == "fake":
        return FakeLLM()
    return OllamaClient(
        config.ollama_host, config.ollama_model, config.ollama_embed_model,
        timeout=config.llm_timeout,
        embed_query_prefix=config.embed_query_prefix,
        embed_document_prefix=config.embed_document_prefix,
    )
