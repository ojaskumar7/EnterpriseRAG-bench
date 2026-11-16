"""Local cross-encoder reranker."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from math import isfinite
from typing import Any

from app.settings import get_settings


class CrossEncoderReranker:
    """Score query-passage pairs with a local sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        predictor: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        requested_device = (device or settings.reranker_device).strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        self.device = requested_device
        self.batch_size = settings.reranker_batch_size if batch_size is None else batch_size
        self.max_length = settings.reranker_max_length if max_length is None else max_length
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self._predictor = predictor
        self._load_lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        with self._load_lock:
            if self._predictor is None:
                from sentence_transformers import CrossEncoder

                self._predictor = CrossEncoder(
                    self.model_name,
                    device=self._resolve_device(),
                    max_length=self.max_length,
                )
        return self._predictor

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        try:
            raw = self._get_predictor().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(f"reranker {self.model_name} failed: {exc}") from exc

        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if values and isinstance(values[0], list):
            values = [value[0] for value in values]
        if len(values) != len(texts):
            raise RuntimeError(
                f"reranker returned {len(values)} scores for {len(texts)} candidates"
            )
        scores = [float(value) for value in values]
        if not all(isfinite(score) for score in scores):
            raise RuntimeError("reranker returned a non-finite score")
        return scores
