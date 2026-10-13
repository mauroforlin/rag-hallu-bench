"""Jev (TypeSafe System One), asked three ways over the same row.

- `jev-choice`: one Choice over the whole response - faithful / unsupported /
  contradicted. Score = 1 - P(faithful), i.e. the probability mass Jev puts on
  either way of being wrong rather than its argmax.
- `jev-noul`: one Noul, "does the response contain anything the documents
  don't back". Score = the noul itself. No confidence field exists for Noul;
  the value is the signal.
- `jev-sentences`: one Choice per response sentence (supports / contradicts /
  says_nothing - the citation_check cookbook's relation and ebm-lens's), all in
  a single request over shared state, as TypeSafe's docs recommend for
  independent questions on the same state. Sentence score = 1 - P(supports);
  response score = the worst sentence. This is the shape that gives a RAG app
  something to act on (which sentence to strike), and on RAGBench it is graded
  against per-sentence labels too.

The question is part of the state on purpose: RAG answers are often elliptical
("Yes, in most patients"), and a verifier that can't see what was asked can't
tell what "yes" is a claim about.

Pinned model, same reason as ebm-lens's typesafe_client: a silent upgrade
behind jev-latest would make numbers from two runs incomparable.
"""
from __future__ import annotations

import os
import time

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result

MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
PRICE_PER_M_INPUT = 0.042  # docs.typesafe.ai/models, output tokens free

_RESPONSE_CRITERIA = {
    "faithful": "Every factual statement in `response` is stated in or directly implied by `documents`.",
    "unsupported": (
        "`response` contains at least one factual statement that `documents` neither state nor imply, "
        "although nothing in `documents` contradicts it."
    ),
    "contradicted": "`response` contains at least one statement that `documents` contradict.",
}
_RESPONSE_INSTRUCTIONS = (
    "`response` was written to answer `question` using only `documents`. "
    "Judge whether `response` stays faithful to `documents`."
)

_NOUL_INSTRUCTIONS = (
    "`response` was written to answer `question` using only `documents`. "
    "Does `response` contain any factual statement that `documents` do not state or directly imply, "
    "or that `documents` contradict?"
)
_NOUL_CRITERIA = {
    "true": "At least one statement in `response` is unsupported by or contradicts `documents`.",
    "false": "Every factual statement in `response` is stated in or directly implied by `documents`.",
}

_SENTENCE_CRITERIA = {
    "supports": "`documents` state the sentence or directly imply that it is true.",
    "contradicts": "`documents` state the opposite of the sentence or imply it is false.",
    "says_nothing": "`documents` do not address what the sentence asserts, either way.",
}


def _sentence_instructions(i: int) -> str:
    return (
        f"`response_sentences[{i}]` is one sentence of `response`, written to answer `question`. "
        f"Read it in the context of `question` and `response`. How do `documents` relate to what "
        f"`response_sentences[{i}]` asserts?"
    )


class _JevBase(Detector):
    workers = 16  # well under the 80 req/s published limit

    def setup(self) -> None:
        from dotenv import load_dotenv
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        load_dotenv()
        key = os.environ.get("TYPESAFE_KEY")
        if not key:
            raise SystemExit("TYPESAFE_KEY is not set (.env)")
        self.client = TypeSafeClient(
            api_key=key,
            retry=RetryPolicy(max_retries=5, http_statuses={429, 500, 502, 503, 504, 529}),
        )

    def _ask(self, ex: Example, questions: dict, state: dict):
        t0 = time.monotonic()
        response = self.client.system_one(state=state, questions=questions, model=MODEL)
        latency = (time.monotonic() - t0) * 1000
        usage = response.usage
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        return response, dict(
            latency_ms=latency,
            input_tokens=tokens_in,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cost_usd=tokens_in * PRICE_PER_M_INPUT / 1e6,
        )

    @staticmethod
    def _state(ex: Example, **extra) -> dict:
        return {"question": ex.question, "documents": ex.documents, "response": ex.response, **extra}


class JevChoice(_JevBase):
    name = "jev-choice"

    def score_one(self, ex: Example) -> Result:
        from typesafe_sdk import Choice

        response, stats = self._ask(
            ex,
            {"faithfulness": Choice(instructions=_RESPONSE_INSTRUCTIONS, criteria=_RESPONSE_CRITERIA)},
            self._state(ex),
        )
        a = response.answers["faithfulness"]
        probs = dict(a.probabilities)
        return Result(
            id=ex.id,
            score=1.0 - probs.get("faithful", 0.0),
            raw={"choice": a.choice, "confidence": a.confidence, "probabilities": probs},
            **stats,
        )


class JevNoul(_JevBase):
    name = "jev-noul"

    def score_one(self, ex: Example) -> Result:
        from typesafe_sdk import Noul

        response, stats = self._ask(
            ex,
            {"hallucinated": Noul(instructions=_NOUL_INSTRUCTIONS, criteria=_NOUL_CRITERIA)},
            self._state(ex),
        )
        noul = response.answers["hallucinated"].noul
        return Result(id=ex.id, score=float(noul), raw={"noul": noul}, **stats)


class JevSentences(_JevBase):
    name = "jev-sentences"

    def score_one(self, ex: Example) -> Result:
        from typesafe_sdk import Choice

        questions = {
            f"s{i}": Choice(instructions=_sentence_instructions(i), criteria=_SENTENCE_CRITERIA)
            for i in range(len(ex.sentences))
        }
        response, stats = self._ask(ex, questions, self._state(ex, response_sentences=ex.sentences))
        per_sentence, raw = [], []
        for i in range(len(ex.sentences)):
            a = response.answers[f"s{i}"]
            probs = dict(a.probabilities)
            per_sentence.append(1.0 - probs.get("supports", 0.0))
            raw.append({"choice": a.choice, "confidence": a.confidence, "probabilities": probs})
        return Result(
            id=ex.id,
            score=max(per_sentence),
            sentence_scores=per_sentence,
            raw={"sentences": raw},
            **stats,
        )


# Prompt-sensitivity ablation: the same judgment asked in other words. Run on
# the core slice only and reported apart from the main ranking.
_PLAIN_INSTRUCTIONS = "Is everything `response` says backed by `documents`?"
_PLAIN_CRITERIA = {
    "yes": "Everything `response` states can be found in `documents` or follows directly from them.",
    "partly": "Some of what `response` states cannot be found in `documents`.",
    "no": "`response` states something that `documents` contradict.",
}
_FLIPPED_NOUL_INSTRUCTIONS = (
    "Could a careful reviewer who reads only `documents` confirm every factual statement in `response`?"
)
_FLIPPED_NOUL_CRITERIA = {
    "true": "Every factual statement in `response` can be confirmed from `documents`.",
    "false": "At least one statement in `response` cannot be confirmed from `documents`, or is contradicted by them.",
}


class JevChoicePlain(_JevBase):
    name = "jev-choice-plain"

    def score_one(self, ex: Example) -> Result:
        from typesafe_sdk import Choice

        response, stats = self._ask(
            ex, {"q": Choice(instructions=_PLAIN_INSTRUCTIONS, criteria=_PLAIN_CRITERIA)}, self._state(ex),
        )
        a = response.answers["q"]
        probs = dict(a.probabilities)
        return Result(id=ex.id, score=1.0 - probs.get("yes", 0.0),
                      raw={"choice": a.choice, "confidence": a.confidence, "probabilities": probs}, **stats)


class JevNoulFlipped(_JevBase):
    name = "jev-noul-flipped"

    def score_one(self, ex: Example) -> Result:
        from typesafe_sdk import Noul

        response, stats = self._ask(
            ex, {"q": Noul(instructions=_FLIPPED_NOUL_INSTRUCTIONS, criteria=_FLIPPED_NOUL_CRITERIA)}, self._state(ex),
        )
        noul = response.answers["q"].noul
        return Result(id=ex.id, score=1.0 - float(noul), raw={"noul": noul}, **stats)
