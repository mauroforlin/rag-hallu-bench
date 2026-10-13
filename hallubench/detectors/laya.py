"""Laya (Convai, Apache-2.0): an open "System One"-style decision model, a
trained decision head on a frozen encoder that takes Jev's request format
(state + choice / noul / score questions) and returns Jev-shaped answers.

It is asked *exactly* the questions `jev-choice` and `jev-noul` ask, over the
same state, so any gap between them is the model, not the prompt. Score =
1 - P(faithful) from the Choice, like `jev-choice`; the Noul answer is kept in
`raw` for comparison.

Checkpoint: `laya-multilingual` with max_len=8192, the one its authors
recommend for long documents. The default English checkpoint stops at 1,024
tokens and would silently cut most RAGBench and HaluBench-CovidQA contexts.
The authors report accuracy varying past ~4,000 tokens, which is part of what
gets measured here. Pinned revision, same reason as Jev's pinned model.
"""
from __future__ import annotations

import time

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result
from hallubench.detectors.jev import (
    _NOUL_CRITERIA,
    _NOUL_INSTRUCTIONS,
    _RESPONSE_CRITERIA,
    _RESPONSE_INSTRUCTIONS,
)

REPO_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"  # convaiinnovations/laya

QUESTIONS = {
    "faithfulness": {"type": "choice", "instructions": _RESPONSE_INSTRUCTIONS, "criteria": _RESPONSE_CRITERIA},
    "hallucinated": {"type": "noul", "instructions": _NOUL_INSTRUCTIONS, "criteria": _NOUL_CRITERIA},
}


class Laya(Detector):
    name = "laya"

    def setup(self) -> None:
        from laya import Router

        from hallubench.detectors.encoders import device

        self.router = Router(device=device(), revision=REPO_REVISION)

    def score_one(self, ex: Example) -> Result:
        state = {"question": ex.question, "documents": ex.documents, "response": ex.response}
        t0 = time.monotonic()
        out = self.router.predict(state, QUESTIONS, model="multilingual", max_len=8192)
        latency = (time.monotonic() - t0) * 1000
        choice = out["answers"]["faithfulness"]
        probs = dict(choice["probabilities"])
        return Result(
            id=ex.id,
            score=1.0 - probs.get("faithful", 0.0),
            latency_ms=latency,
            raw={"choice": choice["choice"], "probabilities": probs,
                 "noul": out["answers"]["hallucinated"]["noul"], "model": out["routing"]["model"]},
        )
