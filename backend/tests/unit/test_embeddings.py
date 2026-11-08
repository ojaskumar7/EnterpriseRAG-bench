"""Tests for embedding providers."""

from __future__ import annotations

import math

import pytest

from app.embeddings.bge import BGEEmbeddingProvider
from app.embeddings.fake import FakeEmbeddingProvider


class FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.dimension = 4
        self.half_called = False

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def half(self) -> FakeSentenceTransformer:
        self.half_called = True
        return self

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((texts, kwargs))
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


def _provider(**overrides: object) -> tuple[BGEEmbeddingProvider, FakeSentenceTransformer]:
    model = FakeSentenceTransformer("fake/model", "cpu")
    provider = BGEEmbeddingProvider(
        model_name="fake/model",
        dimension=4,
        device="cpu",
        batch_size=8,
        query_instruction="QUERY: ",
        model=model,
        **overrides,
    )
    return provider, model


def test_documents_do_not_get_query_instruction() -> None:
    provider, model = _provider()

    vectors = provider.embed_documents(["alpha", "beta"])

    assert len(vectors) == 2
    texts, kwargs = model.calls[0]
    assert texts == ["alpha", "beta"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8
    assert kwargs["show_progress_bar"] is False


def test_query_gets_instruction() -> None:
    provider, model = _provider()

    vector = provider.embed_query("rotate a key")

    assert len(vector) == 4
    texts, _ = model.calls[0]
    assert texts == ["QUERY: rotate a key"]


def test_empty_document_batch_does_not_load_model() -> None:
    provider = BGEEmbeddingProvider(
        model_name="never-loaded",
        dimension=4,
        device="cpu",
    )

    assert provider.embed_documents([]) == []


def test_dimension_mismatch_is_rejected() -> None:
    model = FakeSentenceTransformer("fake/model", "cpu")

    with pytest.raises(ValueError, match="dimension"):
        BGEEmbeddingProvider(
            model_name="fake/model",
            dimension=1024,
            device="cpu",
            model=model,
        )


def test_float16_converts_the_model() -> None:
    model = FakeSentenceTransformer("fake/model", "cpu")

    provider = BGEEmbeddingProvider(
        model_name="fake/model",
        dimension=4,
        device="cpu",
        dtype="float16",
        model=model,
    )

    assert provider.dtype == "float16"
    assert model.half_called is True


def test_invalid_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="dtype"):
        BGEEmbeddingProvider(model_name="fake/model", dimension=4, device="cpu", dtype="int8")


def test_auto_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embeddings.bge.torch.cuda.is_available", lambda: True)

    assert BGEEmbeddingProvider._resolve_device("auto") == "cuda"


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embeddings.bge.torch.cuda.is_available", lambda: False)

    assert BGEEmbeddingProvider._resolve_device("auto") == "cpu"


def test_explicit_cuda_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embeddings.bge.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        BGEEmbeddingProvider._resolve_device("cuda")


def test_fake_provider_is_deterministic_and_normalized() -> None:
    provider = FakeEmbeddingProvider(dimension=16)

    first = provider.embed_query("same text")
    second = provider.embed_query("same text")
    other = provider.embed_query("other text")

    assert first == second
    assert first != other
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-9)
