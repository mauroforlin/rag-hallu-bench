"""General-purpose LLMs used as hallucination judges, through OpenRouter.

Two ways of getting a number out of the same kind of model, because the
interesting question about Jev is not only "does it judge better" but "is its
number more trustworthy", and those need separating:

- `judge-*` (self-reported): the model answers in JSON with a verdict and
  "confidence" it writes itself - the common LLM-as-judge setup, and the one
  ebm-lens compared Jev against on SciFact. Score = confidence if the verdict
  is hallucinated, else 1 - confidence.
- `judge-logprob-*` (computed): the model answers one token, yes or no, and
  the score is P(no) renormalised over those two tokens from the API's
  logprobs. Same idea as Jev's computed distribution, on an open model that
  exposes logprobs. If Jev's calibration edge over `judge-*` shrinks against
  this, the edge was "probabilities vs prose", not anything Jev-specific.

The rubric is the same three-way definition the Jev detectors use, so a gap
between them is the judge, not the question.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import httpx

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result
from hallubench.text import render_documents

_URL = "https://openrouter.ai/api/v1/chat/completions"

_RUBRIC = (
    "A response is FAITHFUL when every factual statement in it is stated in or directly implied by "
    "the documents. It is HALLUCINATED when it contains at least one factual statement the documents "
    "neither state nor imply, or one the documents contradict. Judge only against the documents, "
    "not against your own medical knowledge: a statement that is true in the world but absent from "
    "the documents is still unsupported."
)

_JSON_SYSTEM = (
    "You check whether an answer produced by a retrieval-augmented system stays faithful to the "
    f"documents it was given. {_RUBRIC} Reply with only a JSON object: "
    '{"verdict": "faithful" | "hallucinated", "confidence": <float 0 to 1>}, where confidence is '
    "your honest estimate that your verdict is correct."
)

_YESNO_SYSTEM = (
    "You check whether an answer produced by a retrieval-augmented system stays faithful to the "
    f"documents it was given. {_RUBRIC} Reply with exactly one word: yes if the response is "
    "faithful, no if it is hallucinated."
)


def _user_prompt(ex: Example) -> str:
    return (
        f"DOCUMENTS:\n{render_documents(ex.documents)}\n\n"
        f"QUESTION:\n{ex.question}\n\n"
        f"RESPONSE:\n{ex.response}"
    )


class _OpenRouterJudge(Detector):
    workers = 12
    model = ""
    # None leaves the model's default reasoning behaviour alone
    reasoning: dict | None = None

    def setup(self) -> None:
        from dotenv import load_dotenv

        load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set (.env)")
        self.http = httpx.Client(
            timeout=120,
            headers={"Authorization": f"Bearer {key}", "X-Title": "rag-hallu-bench"},
        )

    def _post(self, body: dict) -> tuple[dict, float]:
        body = {"model": self.model, "temperature": 0, "usage": {"include": True}, **body}
        if self.reasoning is not None:
            body["reasoning"] = self.reasoning
        delay = 2.0
        for attempt in range(6):
            t0 = time.monotonic()
            r = self.http.post(_URL, json=body)
            latency = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                if "error" not in data:
                    return data, latency
                err = data["error"]
            else:
                err = f"{r.status_code} {r.text[:200]}"
            retryable = r.status_code in (200, 408, 429) or r.status_code >= 500
            if attempt == 5 or not retryable:
                raise RuntimeError(str(err))
            time.sleep(delay * (1 + random.random()))
            delay *= 2
        raise RuntimeError("unreachable")

    @staticmethod
    def _stats(data: dict, latency: float) -> dict:
        usage = data.get("usage") or {}
        return dict(
            latency_ms=latency,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            cost_usd=usage.get("cost"),
        )


class SelfReportedJudge(_OpenRouterJudge):
    """*provider* pins one host, as for the logprob judge, so Qwen read both
    ways is the same deployment and only the way its number is read differs."""

    def __init__(self, name: str, model: str, reasoning: dict | None = None, provider: str | None = None):
        self.name, self.model, self.reasoning, self.provider = name, model, reasoning, provider

    def score_one(self, ex: Example) -> Result:
        body = {
            "messages": [
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user", "content": _user_prompt(ex)},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True}
        data, latency = self._post(body)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            # Claude's safety filter declines some virology passages; the row
            # stays unscored and counts against the judge's coverage
            raise RuntimeError("refused by the provider's content filter")
        text = choice["message"]["content"] or ""
        # the first JSON object only: some models append a second one or prose
        start = text.find("{")
        parsed, _ = json.JSONDecoder().raw_decode(text[start if start >= 0 else 0:])
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("faithful", "hallucinated"):
            raise ValueError(f"unexpected verdict {verdict!r}")
        conf = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        score = conf if verdict == "hallucinated" else 1.0 - conf
        return Result(
            id=ex.id, score=score, raw={"verdict": verdict, "confidence": conf, "provider": data.get("provider")},
            **self._stats(data, latency),
        )


class LogprobJudge(_OpenRouterJudge):
    """*system* and *hallucinated_token* exist for the prompt-sensitivity
    ablation (detectors registered with family "ablation"): the same model
    asked in other words, including with the question flipped so "yes" means
    hallucinated. The main detector keeps the defaults."""

    def __init__(self, name: str, model: str, provider: str,
                 system: str | None = None, hallucinated_token: str = "no"):
        self.name, self.model, self.provider = name, model, provider
        self.system = system or _YESNO_SYSTEM
        self.hallucinated_token = hallucinated_token

    def score_one(self, ex: Example) -> Result:
        data, latency = self._post({
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": _user_prompt(ex)},
            ],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 10,
            # One provider, no fallbacks: OpenRouter otherwise spreads requests
            # over hosts serving different quantizations (and one that had
            # retired the model), so rows would be judged by different models.
            "provider": {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True},
        })
        content = data["choices"][0].get("logprobs", {}).get("content") or []
        if not content:
            raise ValueError("provider returned no logprobs")
        p_yes = p_no = 0.0
        for cand in content[0].get("top_logprobs", []):
            tok = cand["token"].strip().lower()
            if tok == "yes":
                p_yes += math.exp(cand["logprob"])
            elif tok == "no":
                p_no += math.exp(cand["logprob"])
        if p_yes + p_no == 0:
            raise ValueError(f"neither yes nor no in top logprobs: {content[0].get('top_logprobs')}")
        p_hal = p_no if self.hallucinated_token == "no" else p_yes
        return Result(
            id=ex.id, score=p_hal / (p_yes + p_no),
            raw={"p_yes": p_yes, "p_no": p_no, "token": content[0].get("token"), "provider": data.get("provider")},
            **self._stats(data, latency),
        )


# Prompt-sensitivity ablation: the main yes/no judge asked in other words.
PLAIN_YESNO_SYSTEM = (
    "Read the documents and the response. Reply with one word: yes if every claim in the response "
    "is backed by the documents, no otherwise."
)
FLIPPED_YESNO_SYSTEM = (
    "You check whether an answer produced by a retrieval-augmented system stays faithful to the "
    f"documents it was given. {_RUBRIC} Does the response contain any statement the documents do not "
    "support? Reply with exactly one word: yes or no."
)
