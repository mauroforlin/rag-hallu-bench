"""Every detector the benchmark knows, by name.

`runs_on` says where it is meant to run: "api" (anywhere with a key), "cpu"
(small enough for a laptop, faster on any GPU), "gpu" (7-8B, needs >= 24 GB).
`core_only` restricts a detector to the stratified core slice (~1000 rows);
the prompt-ablation variants use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hallubench.detectors.base import Detector


@dataclass(frozen=True)
class Spec:
    factory: Callable[[], Detector]
    runs_on: str
    family: str
    core_only: bool = False
    datasets: tuple[str, ...] | None = None
    # "request": latency_ms is the wait for one check (API calls, and Laya /
    # BioDecision, which score one row per call). "batched": batch time divided
    # by batch size on the GPU, i.e. throughput, not what one user waits.
    latency: str = "request"


def _jev(cls_name: str):
    def make():
        from hallubench.detectors import jev
        return getattr(jev, cls_name)()
    return make


def _judge(name: str, model: str, reasoning: dict | None = None, provider: str | None = None):
    def make():
        from hallubench.detectors.llm_judge import SelfReportedJudge
        return SelfReportedJudge(name, model, reasoning, provider)
    return make


def _logprob_judge(name: str, model: str, provider: str):
    def make():
        from hallubench.detectors.llm_judge import LogprobJudge
        return LogprobJudge(name, model, provider)
    return make


def _logprob_variant(name: str, system_attr: str, hallucinated_token: str):
    def make():
        from hallubench.detectors import llm_judge
        return llm_judge.LogprobJudge(name, "qwen/qwen3-235b-a22b-2507", "google-vertex",
                                      getattr(llm_judge, system_attr), hallucinated_token)
    return make


def _encoder(kind: str, *args):
    def make():
        if kind == "hhem":
            from hallubench.detectors.encoders import HHEM
            return HHEM()
        if kind == "minicheck":
            from hallubench.detectors.encoders import MiniCheck
            return MiniCheck(*args)
        if kind == "alignscore":
            from hallubench.detectors.encoders import AlignScore
            return AlignScore()
        if kind == "laya":
            from hallubench.detectors.laya import Laya
            return Laya()
        from hallubench.detectors.lettuce import LettuceDetect
        return LettuceDetect(*args)
    return make


def _vllm(cls_name: str):
    def make():
        from hallubench.detectors import generative
        return getattr(generative, cls_name)()
    return make


def _precomputed(name: str, field: str):
    def make():
        from hallubench.detectors.precomputed import Precomputed
        return Precomputed(name, field)
    return make


def _cloud(cls_name: str):
    def make():
        from hallubench.detectors import cloud
        return getattr(cloud, cls_name)()
    return make


def _baseline(cls_name: str):
    def make():
        from hallubench.detectors import baselines
        return getattr(baselines, cls_name)()
    return make


_RAGBENCH = ("ragbench-pubmedqa", "ragbench-covidqa")

REGISTRY: dict[str, Spec] = {
    # Jev
    "jev-choice": Spec(_jev("JevChoice"), "api", "jev"),
    "jev-noul": Spec(_jev("JevNoul"), "api", "jev"),
    "jev-sentences": Spec(_jev("JevSentences"), "api", "jev"),
    # general LLMs as judges
    "judge-gemini-2.5-flash": Spec(
        _judge("judge-gemini-2.5-flash", "google/gemini-2.5-flash", {"enabled": False}), "api", "llm-judge"),
    "judge-gpt-6-luna": Spec(
        _judge("judge-gpt-6-luna", "openai/gpt-6-luna"), "api", "llm-judge"),
    "judge-gpt-5.6-luna": Spec(
        _judge("judge-gpt-5.6-luna", "openai/gpt-5.6-luna"), "api", "llm-judge"),
    "judge-qwen3-235b": Spec(
        _judge("judge-qwen3-235b", "qwen/qwen3-235b-a22b-2507", provider="google-vertex"), "api", "llm-judge"),
    "judge-logprob-qwen3-235b": Spec(
        _logprob_judge("judge-logprob-qwen3-235b", "qwen/qwen3-235b-a22b-2507", "google-vertex"), "api", "llm-judge"),
    # reasoning switched on
    "judge-claude-haiku-5.5-thinking": Spec(
        _judge("judge-claude-haiku-5.5-thinking", "anthropic/claude-haiku-5.5", {"enabled": True}), "api", "llm-judge"),
    # RAGBench only (real RAG answers), where the judges differ most
    "judge-gemini-3.8-flash-thinking": Spec(
        _judge("judge-gemini-3.8-flash-thinking", "google/gemini-3.8-flash", {"enabled": True}), "api",
        "llm-judge", datasets=_RAGBENCH),
    # small open-weight verifiers
    "hhem": Spec(_encoder("hhem"), "cpu", "open-verifier", latency="batched"),
    "minicheck-ft5": Spec(_encoder("minicheck", "ft5"), "cpu", "open-verifier", latency="batched"),
    "minicheck-deberta": Spec(_encoder("minicheck", "deberta"), "cpu", "open-verifier", latency="batched"),
    "alignscore-large": Spec(_encoder("alignscore"), "cpu", "open-verifier", latency="batched"),
    "lettucedetect-large": Spec(_encoder("lettuce", "large"), "cpu", "open-verifier"),
    "laya": Spec(_encoder("laya"), "cpu", "open-system-one"),
    # open-weight 4-8B judges
    "bespoke-minicheck-7b": Spec(_vllm("BespokeMiniCheck"), "gpu", "open-judge", latency="batched"),
    "lynx-8b": Spec(_vllm("Lynx"), "gpu", "open-judge", latency="batched"),
    "granite-guardian-8b": Spec(_vllm("GraniteGuardian"), "gpu", "open-judge", latency="batched"),
    "biodecision-v2-4b": Spec(_vllm("BioDecision"), "gpu", "open-system-one"),
    "meddecider-4b": Spec(_vllm("MedDecider4B"), "gpu", "open-system-one"),
    "meddecider-9b": Spec(_vllm("MedDecider9B"), "gpu", "open-system-one"),
    "meddecider-27b": Spec(_vllm("MedDecider27B"), "gpu", "open-system-one"),
    "meddecider-31b": Spec(_vllm("MedDecider31B"), "gpu", "open-system-one"),
    # prompt-sensitivity ablation: same models, other wordings, core slice only
    "jev-choice-plain": Spec(_jev("JevChoicePlain"), "api", "ablation", core_only=True),
    "jev-noul-flipped": Spec(_jev("JevNoulFlipped"), "api", "ablation", core_only=True),
    "judge-logprob-qwen3-235b-plain": Spec(
        _logprob_variant("judge-logprob-qwen3-235b-plain", "PLAIN_YESNO_SYSTEM", "no"), "api", "ablation",
        core_only=True),
    "judge-logprob-qwen3-235b-flipped": Spec(
        _logprob_variant("judge-logprob-qwen3-235b-flipped", "FLIPPED_YESNO_SYSTEM", "yes"), "api", "ablation",
        core_only=True),
    # free, shipped with RAGBench
    "ragas-faithfulness": Spec(_precomputed("ragas-faithfulness", "ragas_faithfulness"), "api", "precomputed",
                               datasets=_RAGBENCH),
    "trulens-groundedness": Spec(_precomputed("trulens-groundedness", "trulens_groundedness"), "api",
                                 "precomputed", datasets=_RAGBENCH),
    # managed grounding checks sold by cloud providers
    "vertex-check-grounding": Spec(_cloud("VertexCheckGrounding"), "api", "cloud"),
    "bedrock-contextual-grounding": Spec(_cloud("BedrockContextualGrounding"), "api", "cloud"),
    # trivial references: surface features only, see baselines.py
    "baseline-length": Spec(_baseline("Length"), "cpu", "baseline"),
    "baseline-novel-words": Spec(_baseline("NovelWords"), "cpu", "baseline"),
}
