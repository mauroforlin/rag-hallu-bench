"""Open-weight 7-8B judges, served with vLLM. They need a GPU with at least
24 GB in bf16 (a Colab L4 or A100), see notebooks/colab_gpu.ipynb. A T4 is
fine for the encoders but too small for these.

Every one of them is read through the token probabilities of its verdict,
not the verdict string, so they produce a continuous score the same way Jev
does:

- `bespoke-minicheck-7b` (Bespoke Labs, CC-BY-NC-4.0): the MiniCheck family's
  top model on LLM-AggreFact. It checks one sentence against the document
  with the reference system/user prompt, and is read as P(Yes) over
  {Yes, No} at the first generated token. The response takes its weakest
  sentence. Score = 1 - that support.
- `lynx-8b` (Patronus Lynx 8B v1.1, CC-BY-NC-4.0): the reference prompt, with
  its reasoning generated in full. The score is P(FAIL) over {PASS, FAIL} at
  the token where the verdict starts. It was trained on PubMedQA and CovidQA
  data, so HaluBench is in-distribution for it, which the report flags.
- `granite-guardian-8b` (IBM Granite Guardian 3.3, Apache-2.0): the
  "groundedness" criterion, non-thinking mode. Score = P(yes) over {yes, no}
  at the token inside <score>...</score>. For Guardian, "yes" means the
  response carries the risk, i.e. it is ungrounded.
"""
from __future__ import annotations

import math
import re
import time

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result
from hallubench.text import render_documents

_MINICHECK_SYSTEM = (
    "Determine whether the provided claim is consistent with the corresponding document. Consistency "
    "in this context implies that all information presented in the claim is substantiated by the "
    "document. If not, it should be considered inconsistent. Please assess the claim's consistency with "
    'the document by responding with either "Yes" or "No".'
)

_LYNX_PROMPT = """
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""


def _binary(top: dict, positive: tuple[str, ...], negative: tuple[str, ...]) -> float | None:
    """P(positive) renormalised over the two labels, from one position's
    top-k logprobs. A token counts for a label when, stripped and lowercased,
    it is a non-empty prefix of it ("FA" for FAIL), since a tokenizer can split
    the label word."""
    p_pos = p_neg = 0.0
    for lp in top.values():
        tok = (lp.decoded_token or "").strip().strip('"').lower()
        if not tok:
            continue
        if any(lab.startswith(tok) for lab in positive):
            p_pos += math.exp(lp.logprob)
        elif any(lab.startswith(tok) for lab in negative):
            p_neg += math.exp(lp.logprob)
    if p_pos + p_neg == 0:
        return None
    return p_pos / (p_pos + p_neg)


class _VLLMDetector(Detector):
    workers = 64  # rows per vLLM call; vLLM schedules its own batches inside
    model_id = ""
    max_model_len = 16384

    def setup(self) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=self.model_id,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            seed=0,
            enable_prefix_caching=True,  # the same documents repeat across sentences
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _chat(self, messages: list[dict], **template_kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )


class BespokeMiniCheck(_VLLMDetector):
    name = "bespoke-minicheck-7b"
    model_id = "bespokelabs/Bespoke-MiniCheck-7B"

    def score_batch(self, batch: list[Example]) -> list[Result]:
        from vllm import SamplingParams

        prompts, index = [], []
        for r, ex in enumerate(batch):
            doc = render_documents(ex.documents)
            for s, sentence in enumerate(ex.sentences):
                prompts.append(self._chat([
                    {"role": "system", "content": _MINICHECK_SYSTEM},
                    {"role": "user", "content": f"Document: {doc}\nClaim: {sentence}"},
                ]))
                index.append((r, s))
        t0 = time.monotonic()
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=20),
                                    use_tqdm=False)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        support = [[0.5] * len(ex.sentences) for ex in batch]
        for (r, s), out in zip(index, outputs):
            p = _binary(out.outputs[0].logprobs[0], ("yes",), ("no",))
            support[r][s] = 0.5 if p is None else p
        return [
            Result(id=ex.id, score=1.0 - min(support[r]),
                   sentence_scores=[1.0 - p for p in support[r]], latency_ms=per_row)
            for r, ex in enumerate(batch)
        ]


class Lynx(_VLLMDetector):
    name = "lynx-8b"
    model_id = "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1"

    _SCORE_KEY = re.compile(r'"SCORE"\s*:\s*"?\s*$')

    def score_batch(self, batch: list[Example]) -> list[Result]:
        from vllm import SamplingParams

        prompts = [
            self._chat([{"role": "user", "content": _LYNX_PROMPT.format(
                question=ex.question, context=render_documents(ex.documents), answer=ex.response)}])
            for ex in batch
        ]
        t0 = time.monotonic()
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=600, logprobs=20),
                                    use_tqdm=False)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        results = []
        for ex, out in zip(batch, outputs):
            gen = out.outputs[0]
            p_fail, text = None, ""
            for tok_id, top in zip(gen.token_ids, gen.logprobs):
                if self._SCORE_KEY.search(text):
                    p_fail = _binary(top, ("fail",), ("pass",))
                    break
                text += top[tok_id].decoded_token or ""
            verdict = "FAIL" if "FAIL" in gen.text.split("SCORE")[-1] else (
                "PASS" if "PASS" in gen.text.split("SCORE")[-1] else None)
            if p_fail is None and verdict is not None:
                p_fail = 1.0 if verdict == "FAIL" else 0.0  # verdict found but not at a scorable token
            results.append(Result(
                id=ex.id, score=p_fail, latency_ms=per_row,
                output_tokens=len(gen.token_ids),
                error=None if p_fail is not None else "no PASS/FAIL verdict in output",
                raw={"verdict": verdict, "text": gen.text[-400:]},
            ))
        return results


class GraniteGuardian(_VLLMDetector):
    name = "granite-guardian-8b"
    model_id = "ibm-granite/granite-guardian-3.3-8b"

    def score_batch(self, batch: list[Example]) -> list[Result]:
        from vllm import SamplingParams

        prompts = [
            self._chat(
                [{"role": "user", "content": ex.question}, {"role": "assistant", "content": ex.response}],
                guardian_config={"criteria_id": "groundedness"},
                documents=[{"doc_id": str(i), "text": d} for i, d in enumerate(ex.documents)],
                think=False,
            )
            for ex in batch
        ]
        t0 = time.monotonic()
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=20, logprobs=20),
                                    use_tqdm=False)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        results = []
        for ex, out in zip(batch, outputs):
            gen = out.outputs[0]
            p_yes, text = None, ""
            for tok_id, top in zip(gen.token_ids, gen.logprobs):
                token = top[tok_id].decoded_token or ""
                # the verdict is the first non-blank token after <score>; the
                # tokenizer sometimes emits the space before it on its own
                if text.rstrip().endswith("<score>") and token.strip():
                    p_yes = _binary(top, ("yes",), ("no",))
                    break
                text += token
            results.append(Result(
                id=ex.id, score=p_yes, latency_ms=per_row,
                error=None if p_yes is not None else "no <score> verdict in output",
                raw={"text": gen.text},
            ))
        return results


class BioDecision(Detector):
    """BioDecision v2 (4B, research-only licence): Qwen3.5-4B trained with
    Together's Tev1 recipe on 1.75M biomedical decisions, i.e. a medical
    "System One" model. It ships its own Jev-compatible server; its `Decider`
    class is loaded from the pinned model repo and used as-is, so requests are
    rendered exactly the way its authors evaluate it, and it gets *the same*
    Choice and Noul questions as `jev-choice` / `jev-noul`.

    Score = 1 - P(faithful) from the Choice; the Noul is kept in `raw`.
    Calibrated with the global temperature its repo ships (what its server uses).

    Contamination: its training pool includes MedHallu's generated split and
    PubMedQA-derived HaluBench sources, so on those two it is in-distribution
    (the report flags it). RAGBench is clean for it.
    """

    name = "biodecision-v2-4b"
    workers = 1  # vLLM is not thread-safe; one row at a time through Decider
    repo = "lighteternal/biodecision-v2-4b"
    revision = "f146e2afbd3c07add8791c74d7d2448d18e68b9a"

    def setup(self) -> None:
        import importlib.util

        from huggingface_hub import hf_hub_download, snapshot_download

        local = snapshot_download(self.repo, revision=self.revision)
        server = hf_hub_download(self.repo, "biodecision_server.py", revision=self.revision)
        spec = importlib.util.spec_from_file_location("biodecision_server", server)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 16k instead of its 8k default: HaluBench-CovidQA passages reach ~9k tokens
        self.decider = module.Decider(local, max_model_len=16384, gpu_memory_utilization=0.85)

    def score_one(self, ex: Example) -> Result:
        from hallubench.detectors.laya import QUESTIONS

        state = {"question": ex.question, "documents": ex.documents, "response": ex.response}
        t0 = time.monotonic()
        out = self.decider.decide(state, QUESTIONS)
        latency = (time.monotonic() - t0) * 1000
        choice = out["answers"]["faithfulness"]
        probs = dict(choice["probabilities"])
        return Result(
            id=ex.id,
            score=1.0 - probs.get("faithful", 0.0),
            latency_ms=latency,
            raw={"choice": choice["choice"], "probabilities": probs, "noul": out["answers"]["hallucinated"]["noul"]},
        )


class _MedDecider(Detector):
    """MedDecider (thelion.ai, CC-BY-NC-4.0): LoRA adapters on Qwen and Gemma
    bases, trained on 35.8k medical decisions to answer Jev-style questions
    with a probability per option, read from the option letters' next-token
    logits and averaged over both option orders. Its `MedDecider` class is
    loaded from the pinned adapter repo and used as-is, like BioDecision's
    server, so prompts are rendered exactly the way its authors run it. Plain
    transformers rather than vLLM, one row at a time.

    It gets `jev-choice`'s question over the same state; each option is the
    criterion as "label: description", the format its model card uses. Score
    = 1 - P(faithful), with the choice temperature its config ships. The Noul
    is not asked: two forward passes per question make it twice the GPU time.

    Precision: bf16 as its authors run it when the GPU holds it; otherwise
    (27B and 31B on a 40 GB A100) the base is loaded in 8-bit with
    bitsandbytes and the adapter is applied unmerged on top, the usual QLoRA
    inference path, instead of merging into quantised weights. Set
    MEDDECIDER_8BIT=0/1 to force either. Each row records which in `raw`, so
    the two are never mixed unnoticed.

    Contamination: none of the five datasets is in its training list, which
    does include NLI and evidence-inference sets (BioNLI, Evidence Inference,
    VitaminC, MultiNLI), so the task is familiar but the rows are not.
    """

    workers = 1
    repo = ""
    revision = ""
    bf16_gb = 0.0  # weights alone; activations and the KV of a 9k-token row come on top

    def _eight_bit(self) -> bool:
        import os

        import torch

        forced = os.environ.get("MEDDECIDER_8BIT")
        if forced is not None:
            return forced == "1"
        return torch.cuda.get_device_properties(0).total_memory < (self.bf16_gb + 12) * 1e9

    def setup(self) -> None:
        import importlib.util
        from pathlib import Path

        from huggingface_hub import snapshot_download

        local = snapshot_download(self.repo, revision=self.revision)
        spec = importlib.util.spec_from_file_location("meddecider_infer", Path(local) / "meddecider_infer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.precision = "int8" if self._eight_bit() else "bf16"
        if self.precision == "int8":
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            class Quantised:
                @staticmethod
                def from_pretrained(base, **kw):
                    kw["device_map"] = {"": 0}
                    return AutoModelForCausalLM.from_pretrained(
                        base, quantization_config=BitsAndBytesConfig(load_in_8bit=True), **kw)

            class Unmerged:
                @staticmethod
                def from_pretrained(model, adapter):
                    peft = PeftModel.from_pretrained(model, adapter)
                    peft.merge_and_unload = lambda: peft
                    return peft

            module.AutoModelForCausalLM, module.PeftModel = Quantised, Unmerged
        self.decider = module.MedDecider(local)

    def score_one(self, ex: Example) -> Result:
        from hallubench.detectors.jev import _RESPONSE_CRITERIA, _RESPONSE_INSTRUCTIONS

        labels = list(_RESPONSE_CRITERIA)
        options = [f"{k}: {v}" for k, v in _RESPONSE_CRITERIA.items()]
        state = {"question": ex.question, "documents": ex.documents, "response": ex.response}
        t0 = time.monotonic()
        out = self.decider.decide(state, _RESPONSE_INSTRUCTIONS, options, qtype="choice")
        latency = (time.monotonic() - t0) * 1000
        probs = {k: out[o] for k, o in zip(labels, options)}
        raw = {"choice": max(probs, key=probs.get), "probabilities": probs, "precision": self.precision}
        if not all(math.isfinite(p) for p in probs.values()):  # int8 overflow shows up as NaN
            return Result(id=ex.id, score=None, latency_ms=latency, error="non-finite probabilities", raw=raw)
        return Result(id=ex.id, score=1.0 - probs["faithful"], latency_ms=latency, raw=raw)


class MedDecider4B(_MedDecider):
    name = "meddecider-4b"
    repo = "lion-ai/MedDecider-4B"
    revision = "570b37090c379ad722186a79a3432a31be463fc9"
    bf16_gb = 9


class MedDecider9B(_MedDecider):
    name = "meddecider-9b"
    repo = "lion-ai/MedDecider-9B"
    revision = "ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3"
    bf16_gb = 19


class MedDecider27B(_MedDecider):
    name = "meddecider-27b"
    repo = "lion-ai/MedDecider-27B"
    revision = "c30e188118471138300c79f2f963cbdac530e658"
    bf16_gb = 56


class MedDecider31B(_MedDecider):
    """On google/gemma-4-31b-it, which is gated: accept its licence on the
    Hub and give Colab an HF_TOKEN secret."""

    name = "meddecider-31b"
    repo = "lion-ai/MedDecider-31B"
    revision = "035f4543edf46202347515cda8201f1457b1d5fb"
    bf16_gb = 62
