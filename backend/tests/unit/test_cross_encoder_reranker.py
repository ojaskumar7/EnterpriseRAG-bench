from __future__ import annotations

import pytest

from app.reranking.cross_encoder import CrossEncoderReranker


class FakePredictor:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": pairs, "kwargs": kwargs})
        return self.scores


def test_scores_pairs_with_injected_predictor() -> None:
    predictor = FakePredictor([0.2, 0.9])
    reranker = CrossEncoderReranker(predictor=predictor, batch_size=8)

    scores = reranker.score_pairs("query", ["a", "b"])

    assert scores == [0.2, 0.9]
    assert predictor.calls[0]["pairs"] == [("query", "a"), ("query", "b")]
    assert predictor.calls[0]["kwargs"]["batch_size"] == 8
    assert predictor.calls[0]["kwargs"]["show_progress_bar"] is False


def test_empty_texts_do_not_call_predictor() -> None:
    predictor = FakePredictor([])
    reranker = CrossEncoderReranker(predictor=predictor)

    assert reranker.score_pairs("query", []) == []
    assert predictor.calls == []


def test_rejects_invalid_settings_and_device() -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(device="mps")
    with pytest.raises(ValueError):
        CrossEncoderReranker(batch_size=0)
    with pytest.raises(ValueError):
        CrossEncoderReranker(max_length=0)


def test_rejects_wrong_score_count_and_non_finite_scores() -> None:
    with pytest.raises(RuntimeError):
        CrossEncoderReranker(predictor=FakePredictor([0.1])).score_pairs("q", ["a", "b"])
    with pytest.raises(RuntimeError):
        CrossEncoderReranker(predictor=FakePredictor([float("nan")])).score_pairs("q", ["a"])
