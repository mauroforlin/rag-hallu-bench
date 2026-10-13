"""The one contract every detector meets.

`score` is P(hallucinated) in [0, 1] - higher means "this response says
something the documents don't back". Each detector maps its native output
onto that direction (HHEM's consistency score becomes 1 - score, Lynx's
PASS/FAIL becomes P(FAIL), ...), and says how in its own docstring. AUROC
doesn't care whether the number is a calibrated probability; the calibration
section of the report does, and skips the precomputed RAGAS/TruLens fractions.

`sentence_scores`, when a detector judges sentence by sentence, is aligned to
`Example.sentences` - for RAGBench that is the dataset's own sentence split, so
sentence-level AUROC compares against its sentence labels one-to-one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hallubench.data import Example


@dataclass
class Result:
    id: str
    score: float | None
    sentence_scores: list[float] | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    raw: dict = field(default_factory=dict)


class Detector:
    name: str = ""
    # concurrent requests for API detectors; batch size for local models
    workers: int = 1

    def setup(self) -> None:
        """Load weights / clients once, before the first row."""

    def score_one(self, ex: Example) -> Result:
        raise NotImplementedError

    def score_batch(self, batch: list[Example]) -> list[Result]:
        """Local models override this to batch on the GPU; API detectors keep
        the default and get parallelism from `workers` threads instead."""
        return [self.score_one(ex) for ex in batch]
