"""Scores RAGBench ships with every row, so they cost nothing to include:
RAGAS faithfulness and TruLens groundedness, both LLM-as-judge pipelines run
by the RAGBench authors. Score = 1 - the published value. RAGBench rows only.
"""
from __future__ import annotations

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result


class Precomputed(Detector):
    workers = 1000

    def __init__(self, name: str, field: str):
        self.name, self.field = name, field

    def score_one(self, ex: Example) -> Result:
        value = ex.meta.get(self.field)
        if value is None:
            return Result(id=ex.id, score=None, error=f"{self.field} not available for {ex.dataset}")
        return Result(id=ex.id, score=1.0 - float(value))
