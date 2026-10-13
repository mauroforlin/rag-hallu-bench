"""Small open-weight verifiers (< 1B parameters). They run on CPU, and much
faster on the Colab GPU.

- `hhem`: Vectara HHEM-2.1-Open (Apache-2.0). Native usage: premise = the
  documents, hypothesis = the response, output = consistency in [0, 1].
  Score = 1 - consistency. Each sentence is also scored as its own
  hypothesis, for the sentence-level table.
- `minicheck-ft5` / `minicheck-deberta`: MiniCheck (Tang et al., EMNLP 2024),
  reimplemented from the reference code at github.com/Liyan06/MiniCheck rather
  than pip-installed, since that package pins old dependencies. The scoring is
  kept as the reference does it: the documents are split with nltk into chunks
  (500 words for Flan-T5, 400 tokens for DeBERTa), each sentence takes its
  maximum support over the chunks, and the response takes its weakest sentence.
  Score = 1 - that support.
- `alignscore-large`: AlignScore (Zha et al., ACL 2023), the RoBERTa-large
  checkpoint, as NVIDIA NeMo Guardrails uses it for its fact-checking rail.
  Reimplemented from github.com/yuh-zha/AlignScore (inference.py) for the
  same reason as MiniCheck. Its documented "nli_sp" mode: the premise is cut
  into chunks of about 350 words at nltk sentence boundaries, each nltk
  sentence of the response takes its highest "aligned" probability over the
  chunks, and the response takes the mean over its sentences.
  Score = 1 - that mean.

Like every encoder verifier, these don't see the question. That is how they
are meant to be used, and it costs them on elliptical answers ("Yes.").
"""
from __future__ import annotations

import time

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result
from hallubench.text import render_documents


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def quadratic_batches(lengths: list[int], budget: float) -> list[list[int]]:
    """Group item indices so that each group's padded attention cost
    (items x longest_length^2) stays under *budget*. A fixed batch size isn't
    safe here: RAGBench premises reach ~7k tokens, and padding a batch of
    sentences to that length is what filled a 16 GB laptop's RAM. Items are
    sorted by length so short ones aren't padded to a long one. A single item
    over budget still gets its own batch, it can't be split."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        longest = max(lengths[i], *(lengths[j] for j in current)) if current else lengths[i]
        if current and (len(current) + 1) * longest ** 2 > budget:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


class HHEM(Detector):
    name = "hhem"
    workers = 8  # rows per call; their (premise, hypothesis) pairs are re-batched by quadratic_batches

    def setup(self) -> None:
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True,
        ).to(device()).eval()

    def score_batch(self, batch: list[Example]) -> list[Result]:
        import torch

        pairs, spans = [], []
        for ex in batch:
            premise = render_documents(ex.documents)
            start = len(pairs)
            pairs.append((premise, ex.response))
            pairs.extend((premise, s) for s in ex.sentences)
            spans.append((start, len(pairs)))
        # ~1.5 T5 tokens per word; the budget is in tokens^2
        lengths = [int(1.5 * (len(p.split()) + len(h.split()))) + 8 for p, h in pairs]
        # 6e7 ran an A100-40GB out of memory on RAGBench's longest premises
        budget = 2e7 if device() == "cuda" else 1.2e7
        consistency = [0.0] * len(pairs)
        t0 = time.monotonic()
        with torch.no_grad():
            for idx in quadratic_batches(lengths, budget):
                try:
                    values = self.model.predict([pairs[i] for i in idx])
                except torch.OutOfMemoryError:
                    # fragmentation can still exhaust the GPU; one pair at a
                    # time is slower but scores the same rows
                    torch.cuda.empty_cache()
                    values = [self.model.predict([pairs[i]])[0] for i in idx]
                for i, value in zip(idx, values):
                    consistency[i] = float(value)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        out = []
        for ex, (a, b) in zip(batch, spans):
            out.append(Result(
                id=ex.id,
                score=1.0 - consistency[a],
                sentence_scores=[1.0 - c for c in consistency[a + 1:b]],
                latency_ms=per_row,
            ))
        return out


def _nltk_sentences(text: str) -> list[str]:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    # same as MiniCheck's sent_tokenize_with_newlines
    out = []
    for block in text.split("\n"):
        out.extend(nltk.sent_tokenize(block))
        out.append("\n")
    return out[:-1]


class MiniCheck(Detector):
    workers = 4  # rows per call; every (chunk, sentence) pair of those rows is one forward batch
    inner_batch = 16

    def __init__(self, variant: str):
        self.variant = variant
        self.name = f"minicheck-{variant}"

    def setup(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

        dev = device()
        if self.variant == "ft5":
            ckpt = "lytang/MiniCheck-Flan-T5-Large"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
            self.max_len, self.chunk_size = 2048, 500
        elif self.variant == "deberta":
            ckpt = "lytang/MiniCheck-DeBERTa-v3-Large"
            self.model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
            self.max_len, self.chunk_size = 2048, 400
        else:
            raise ValueError(self.variant)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
        self.model = self.model.to(dev).eval()
        self.torch = torch

    def _length(self, sentence: str) -> int:
        if self.variant == "ft5":
            return len(sentence.split())
        return len(self.tokenizer(sentence, add_special_tokens=False, truncation=True,
                                  max_length=self.max_len)["input_ids"])

    def _chunks(self, doc: str) -> list[str]:
        chunks, current, count = [], [], 0
        for sent in _nltk_sentences(doc) or [""]:
            n = self._length(sent)
            if current and count + n > self.chunk_size:
                chunks.append(" ".join(current))
                current, count = [sent], n
            else:
                current.append(sent)
                count += n
        if current:
            chunks.append(" ".join(current))
        return [c.replace(" \n ", "\n").strip() for c in chunks if c.strip()] or [""]

    def _support(self, texts: list[str]) -> list[float]:
        torch = self.torch
        probs: list[float] = []
        for i in range(0, len(texts), self.inner_batch):
            mini = texts[i:i + self.inner_batch]
            if self.variant == "ft5":
                mini = ["predict: " + t for t in mini]
            enc = self.tokenizer(mini, max_length=self.max_len, truncation=True, padding=True,
                                 return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                if self.variant == "ft5":
                    dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=self.model.device)
                    logits = self.model(**enc, decoder_input_ids=dec).logits.squeeze(1)
                    p = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1]  # 3 = no, 209 = support
                else:
                    p = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
        return probs

    def score_batch(self, batch: list[Example]) -> list[Result]:
        eos = self.tokenizer.eos_token
        texts, index = [], []  # index[k] = (row, sentence) for texts[k]
        for r, ex in enumerate(batch):
            chunks = self._chunks(render_documents(ex.documents))
            for s, sentence in enumerate(ex.sentences):
                for chunk in chunks:
                    texts.append(f"{chunk}{eos}{sentence}")
                    index.append((r, s))
        t0 = time.monotonic()
        probs = self._support(texts)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        best = [[0.0] * len(ex.sentences) for ex in batch]
        for (r, s), p in zip(index, probs):
            best[r][s] = max(best[r][s], p)
        return [
            Result(
                id=ex.id,
                score=1.0 - min(best[r]),
                sentence_scores=[1.0 - p for p in best[r]],
                latency_ms=per_row,
            )
            for r, ex in enumerate(batch)
        ]


class AlignScore(Detector):
    name = "alignscore-large"
    workers = 8  # rows per call; their (chunk, sentence) pairs are re-batched by quadratic_batches
    revision = "8509e78d25bb914939fc585c626500c9b2944249"

    def setup(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, RobertaModel

        ckpt = hf_hub_download("yzha/AlignScore", "AlignScore-large.ckpt", revision=self.revision)
        # a Lightning checkpoint: its hyper-parameters are pickled, hence weights_only=False
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.base = RobertaModel.from_pretrained("roberta-large")
        # strict: every weight, the pooler included, must come from the checkpoint.
        # position_ids is a constant buffer older transformers versions saved.
        self.base.load_state_dict({k[len("base_model."):]: v for k, v in state.items()
                                   if k.startswith("base_model.") and not k.endswith("position_ids")},
                                  strict=True)
        self.tri = torch.nn.Linear(self.base.config.hidden_size, 3)
        self.tri.load_state_dict({k[len("tri_layer."):]: v for k, v in state.items() if k.startswith("tri_layer.")})
        self.base, self.tri = self.base.to(device()).eval(), self.tri.to(device()).eval()
        self.torch = torch

    @staticmethod
    def _chunks(premise: str) -> list[str]:
        import nltk

        _nltk_sentences("")  # makes sure punkt_tab is there
        sents = nltk.sent_tokenize(premise) or [""]
        n_chunk = len(premise.strip().split()) // 350 + 1
        per = max(len(sents) // n_chunk, 1)
        return [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)]

    def _aligned(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch = self.torch
        enc_lengths = [len(self.tokenizer(c, h, truncation="only_first", max_length=512)["input_ids"])
                       for c, h in pairs]
        out = [0.0] * len(pairs)
        for idx in quadratic_batches(enc_lengths, 4e6 if device() == "cuda" else 1e6):
            enc = self.tokenizer([pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                                 truncation="only_first", max_length=512, padding=True,
                                 return_tensors="pt").to(device())
            with torch.no_grad():
                pooled = self.base(**enc).pooler_output
                p = torch.softmax(self.tri(pooled).float(), dim=-1)[:, 0]
            for i, v in zip(idx, p.cpu().tolist()):
                out[i] = v
        return out

    def score_batch(self, batch: list[Example]) -> list[Result]:
        import nltk

        pairs, layout = [], []
        for ex in batch:
            chunks = self._chunks(render_documents(ex.documents))
            hypos = nltk.sent_tokenize(ex.response) or [ex.response]
            layout.append((len(pairs), len(chunks), len(hypos)))
            pairs.extend((c, h) for c in chunks for h in hypos)
        t0 = time.monotonic()
        probs = self._aligned(pairs)
        per_row = (time.monotonic() - t0) * 1000 / len(batch)
        out = []
        for ex, (start, n_c, n_h) in zip(batch, layout):
            best = [max(probs[start + c * n_h + h] for c in range(n_c)) for h in range(n_h)]
            out.append(Result(id=ex.id, score=1.0 - sum(best) / n_h, latency_ms=per_row,
                              raw={"sentences": n_h, "chunks": n_c}))
        return out
