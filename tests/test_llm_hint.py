from jobsearch.config import Config
from jobsearch.matching.llm import llm_hint


def test_llm_hint_for_ollama():
    assert "ollama serve" in llm_hint(Config(llm_backend="ollama"))


def test_llm_hint_for_llamacpp():
    assert "llama-server" in llm_hint(Config(llm_backend="llamacpp"))


def test_llm_hint_for_anthropic():
    assert "ANTHROPIC_API_KEY" in llm_hint(Config(llm_backend="anthropic"))


def test_llm_hint_empty_for_unknown_backend():
    assert llm_hint(Config(llm_backend="fake")) == ""
