"""LettuceDetect (Kovács & Recski 2025, MIT): a ModernBERT token classifier
trained on RAGTruth that marks which tokens of the answer are unsupported by
the context. It is the only detector here built to localise errors inside an
answer rather than give a verdict on it.

Score = the highest hallucination probability over the answer's tokens: a
response is as suspicious as its most suspicious token. RAGTruth has no medical
data, so every dataset here is out of domain for it.
"""
from __future__ import annotations

import time

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result


class LettuceDetect(Detector):
    def __init__(self, size: str = "large"):
        self.size = size
        self.name = f"lettucedetect-{size}"

    def setup(self) -> None:
        from lettucedetect.models.inference import HallucinationDetector

        self.detector = HallucinationDetector(
            method="transformer",
            model_path=f"KRLabsOrg/lettucedect-{self.size}-modernbert-en-v1",
        )

    def score_one(self, ex: Example) -> Result:
        t0 = time.monotonic()
        tokens = self.detector.predict(
            context=ex.documents, question=ex.question, answer=ex.response, output_format="tokens",
        )
        latency = (time.monotonic() - t0) * 1000
        probs = [float(t["prob"]) for t in tokens]
        return Result(
            id=ex.id,
            score=max(probs) if probs else 0.0,
            latency_ms=latency,
            raw={"n_tokens": len(probs), "n_flagged": sum(1 for t in tokens if t["pred"] == 1)},
        )
