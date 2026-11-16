"""Reranker interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]: ...
