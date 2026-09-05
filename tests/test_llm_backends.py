"""Backend selection, and llama.cpp's OpenAI-compatible request/response shape.

llama.cpp and Ollama both run locally but speak different protocols, so this
guards the one thing that is easy to get wrong: the URL path and JSON shape
each expects.
"""

import httpx
import pytest

from jobsearch.config import Config
from jobsearch.matching.llm import build_embedder, build_llm
from jobsearch.matching.llm.fake import FakeLLM
from jobsearch.matching.llm.llamacpp import LlamaCppClient
from jobsearch.matching.llm.ollama import OllamaClient


def test_build_llm_routes_by_backend():
    assert build_llm(Config(llm_backend="ollama")).name.startswith("ollama:")
    assert build_llm(Config(llm_backend="llamacpp")).name.startswith("llamacpp:")
    assert isinstance(build_llm(Config(llm_backend="fake")), FakeLLM)


def test_build_embedder_follows_the_chat_backend():
    # Anthropic has no embedding endpoint, so it falls back to Ollama.
    assert isinstance(build_embedder(Config(llm_backend="anthropic")), OllamaClient)
    # llama.cpp serves both from the same server.
    assert isinstance(build_embedder(Config(llm_backend="llamacpp")), LlamaCppClient)
    # fake must stay fake end-to-end, or a "fake everything" test run would
    # silently reach for a real Ollama server for embeddings only.
    assert isinstance(build_embedder(Config(llm_backend="fake")), FakeLLM)


def test_build_embedder_passes_the_distinct_embed_model_for_llamacpp():
    # Behind llama-swap, chat and embeddings are different model ids -- if
    # build_embedder forgot to pass embed_model, embeddings would silently be
    # requested against the chat model's name and 404.
    config = Config(
        llm_backend="llamacpp",
        llamacpp_model="qwen3.6-27b",
        llamacpp_embed_model="nomic-embed-text-v1.5",
    )
    embedder = build_embedder(config)
    assert embedder.embed_model == "nomic-embed-text-v1.5"
    assert build_llm(config).model == "qwen3.6-27b"


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


def test_llamacpp_complete_json_uses_openai_chat_shape(monkeypatch):
    captured = {}

    def fake_post(url, timeout=None, json=None):
        captured["url"], captured["json"] = url, json
        return _FakeResponse({"choices": [{"message": {"content": '{"score": 77}'}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    client = LlamaCppClient("http://desktop:8080", model="qwen2.5")
    result = client.complete_json("sys", "prompt")

    assert result == {"score": 77}
    assert captured["url"] == "http://desktop:8080/v1/chat/completions"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["json"]["model"] == "qwen2.5"


def test_llamacpp_uses_different_models_for_chat_and_embed(monkeypatch):
    """The bug this guards against: behind llama-swap, `model` routes the
    request, so sending the chat model's name to /v1/embeddings 404s even
    though the server is up and the endpoint exists."""
    seen_models = []

    def fake_post(url, timeout=None, json=None):
        seen_models.append((url, json["model"]))
        if url.endswith("/v1/chat/completions"):
            return _FakeResponse({"choices": [{"message": {"content": "{}"}}]})
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.1]}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    client = LlamaCppClient("http://x:8080", model="qwen3.6-27b", embed_model="nomic-embed-text-v1.5")
    client.complete_json("s", "p")
    client.embed(["a"])

    assert seen_models == [
        ("http://x:8080/v1/chat/completions", "qwen3.6-27b"),
        ("http://x:8080/v1/embeddings", "nomic-embed-text-v1.5"),
    ]


def test_llamacpp_embed_model_defaults_to_chat_model_when_unset():
    client = LlamaCppClient("http://x:8080", model="qwen3.6-27b")
    assert client.embed_model == "qwen3.6-27b"


def test_llamacpp_names_reasoning_truncation_instead_of_empty_response(monkeypatch):
    """Reproduces a real failure: a Qwen3-family reasoning model wrote its
    whole chain of thought to reasoning_content and hit max_tokens before ever
    writing to content. The generic "empty response" salvage error does not
    say why; this one does."""
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _FakeResponse({"choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "",
                        "reasoning_content": "Let me think step by step..." * 20},
        }]}),
    )
    with pytest.raises(Exception, match="reasoning"):
        LlamaCppClient("http://x:8080", model="qwen3.6-27b").complete_json("s", "p", max_tokens=300)


def test_llamacpp_surfaces_the_servers_error_body_not_just_the_status_line(monkeypatch):
    """A bare str(HTTPStatusError) is just "400 Bad Request for url ...".
    llama.cpp's actual diagnosis (e.g. a context-size error) lives in the
    response body, which must reach the raised LLMError or every failure
    looks identical regardless of cause."""
    request = httpx.Request("POST", "http://x:8080/v1/embeddings")
    response = httpx.Response(
        400, request=request,
        json={"error": {"message": "input (3167 tokens) is larger than the max "
                                    "context size (2048 tokens)"}},
    )

    def fake_post(url, timeout=None, json=None):
        raise httpx.HTTPStatusError("400 Bad Request", request=request, response=response)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(Exception, match="3167 tokens"):
        LlamaCppClient("http://x:8080").embed(["a"])


def test_llamacpp_normal_truncation_without_reasoning_still_salvages_or_fails_plainly(monkeypatch):
    # A non-reasoning model hitting length with partial-but-parseable JSON
    # must not be misdiagnosed as a reasoning failure.
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _FakeResponse({"choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": '{"score": 42}'},
        }]}),
    )
    assert LlamaCppClient("http://x:8080").complete_json("s", "p") == {"score": 42}


def test_llamacpp_complete_json_salvages_prose_wrapped_output(monkeypatch):
    # response_format is a hint the model can ignore; the salvage parser is
    # what actually guarantees usable output.
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: _FakeResponse(
            {"choices": [{"message": {"content": 'Sure, here: {"score": 5} enjoy'}}]}
        ),
    )
    assert LlamaCppClient("http://x:8080").complete_json("s", "p") == {"score": 5}


def test_llamacpp_embed_uses_openai_embeddings_shape_and_reorders_by_index(monkeypatch):
    def fake_post(url, timeout=None, json=None):
        assert url == "http://x:8080/v1/embeddings"
        assert json["input"] == ["a", "b"]
        # Server returns them out of order; the client must not assume order.
        return _FakeResponse({"data": [
            {"index": 1, "embedding": [0.2]},
            {"index": 0, "embedding": [0.1]},
        ]})

    monkeypatch.setattr(httpx, "post", fake_post)
    vectors = LlamaCppClient("http://x:8080").embed(["a", "b"])
    assert vectors == [[0.1], [0.2]]


def test_llamacpp_embed_error_names_the_missing_flag(monkeypatch):
    def fake_post(url, timeout=None, json=None):
        return _FakeResponse({}, status=501)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(Exception, match="--embedding"):
        LlamaCppClient("http://x:8080").embed(["a"])
