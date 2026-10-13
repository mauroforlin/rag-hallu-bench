"""Public medical hallucination benchmarks, flattened into one row shape.

Every row is (question, retrieved documents, a response) plus a gold label:
1 = the response says something the documents don't back, 0 = it doesn't.
They differ in where the response and the label come from, which is the
point of using several. Three are in the study; ExpertQA's loader is kept but
switched off (see INCLUDE_EXPERTQA):

- RAGBench (Friel et al. 2024), PubMedQA + CovidQA-RAG configs. Responses
  written by real LLMs (GPT-3.5, Claude 3 Haiku) over retrieved passages - the
  closest thing here to "the final output of a RAG system". Labels are
  sentence-level and come from GPT-4-turbo / GPT-4o, not people: a detector that
  thinks like GPT-4 is favoured, and that is reported, not hidden.
- HaluBench (Ravi et al. 2024, Patronus), PubMedQA + CovidQA subsets. Answers
  perturbed to sound plausible while being unfaithful. Lynx was trained on these
  two sources, so its numbers here are in-distribution.
- MedHallu (Pandit et al., EMNLP 2025), pqa_labeled. Each PubMedQA question gives
  two rows: its gold long answer (label 0) and a generated hallucinated answer
  (label 1), with a difficulty level. Note that the gold long answer is the abstract's *conclusion*, and the
  Knowledge passages are the rest of the abstract, so a gold answer can carry a
  conclusion the passages never state. A strict groundedness detector is
  supposed to flag that. MedHallu partly grades factual correctness, not only
  grounding.
- ExpertQA (Malaviya et al., NAACL 2024), Healthcare / Medicine field, the two
  retrieve-and-read systems (GPT-4 over Google results / over the Sphere
  corpus, citing a passage per sentence). Each row is one cited claim and the
  passages it cites, labelled by a medical professional: "Complete" support is
  0, "Partial or Incomplete" is 1 (the paper's own wording for one rating).
  Claims with no evidence attached ("Missing") or a broken link ("N/A") are
  left out, since there is nothing to check them against. The only dataset
  here with human labels; one expert per claim (the authors report >85%
  agreement on a re-labelled medical sample). Experts also rated each claim's
  medical correctness, kept in `meta` to compare "supported" with "true".

`build()` writes data/examples.jsonl once (sampling with a fixed seed);
everything else reads that file, so every detector is graded on byte-identical
rows wherever it runs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hallubench.text import split_sentences

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = ROOT / "data" / "examples.jsonl"

SEED = 20261011
RAGBENCH_PUBMEDQA_ROWS = 1000  # of 2450 in the test split; covidqa's 246 are all used
CORE_SHARE = 0.19  # stratified slice every detector runs on, including the paid ones
DEV_SHARE = 0.3  # thresholds are chosen on dev, reported on test

# Pinned Hub revisions: an upstream re-upload would otherwise change the rows
# under every number in the report without anything here noticing.
REVISIONS = {
    "galileo-ai/ragbench": "97808f3e5fd16ede40bbff6c2949af8139b2eb7b",
    "PatronusAI/HaluBench": "5966a87929f51c204ab3cbef986b449495cc97b6",
    "UTAustin-AIHealth/MedHallu": "515060458a945c633debc6fd5baac7764416b724",
}
EXPERTQA_URL = ("https://raw.githubusercontent.com/chaitanyamalaviya/ExpertQA/"
                "1aa7ba81bd4c083c12ef1aa2fee338f6cb60fa87/data/r2_compiled_anon.jsonl")
EXPERTQA_PATH = ROOT / "data" / "expertqa" / "r2_compiled_anon.jsonl"

DATASETS = (
    "ragbench-pubmedqa",
    "ragbench-covidqa",
    "halubench-pubmedqa",
    "halubench-covidqa",
    "medhallu",
)
# Off until its evidence is cleaned: the stored passages are fragments of the
# pages the experts read, and some labels track medical correctness rather
# than support by the cited page. `_expertqa()` stays ready for when it is.
INCLUDE_EXPERTQA = False
if INCLUDE_EXPERTQA:
    DATASETS = (*DATASETS, "expertqa-med")
# Ranked on these. ExpertQA is reported beside them, not averaged in: its
# stored evidence is an 80-190 word fragment of what the experts judged (they
# could open the full page), so its labels can't be reproduced from the text a
# detector sees. Every detector here lands near chance on it, as every model
# does on LLM-AggreFact's ExpertQA column (~58-61% balanced accuracy).
PRIMARY_DATASETS = DATASETS[:5]


@dataclass
class Example:
    id: str
    dataset: str
    split: str  # "dev" | "test"
    core: bool
    question: str
    documents: list[str]
    response: str
    label: int  # 1 = hallucinated
    sentences: list[str]
    sentence_labels: list[int] | None = None  # RAGBench only
    meta: dict = field(default_factory=dict)


def _split_for(group_key: str) -> str:
    h = int(hashlib.sha256(group_key.encode()).hexdigest(), 16)
    return "dev" if (h % 1000) / 1000 < DEV_SHARE else "test"


def _ragbench(config: str, limit: int | None, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("galileo-ai/ragbench", config, split="test", revision=REVISIONS["galileo-ai/ragbench"])
    rows = list(ds)
    if limit is not None and limit < len(rows):
        by_label = defaultdict(list)
        for r in rows:
            by_label[bool(r["adherence_score"])].append(r)
        rows = []
        for lab, group in sorted(by_label.items()):
            k = round(limit * len(group) / len(ds))
            rows.extend(rng.sample(group, k))
    out = []
    name = f"ragbench-{config}"
    for r in rows:
        keys = [k for k, _ in r["response_sentences"]]
        sentences = [s for _, s in r["response_sentences"]]
        unsupported = set(r["unsupported_response_sentence_keys"] or [])
        # RAGBench's PubMedQA config answers each question twice (GPT-3.5 and
        # Claude 3 Haiku) under one id, so the generator is part of the row id.
        # Both answers share a split and a bootstrap group, as MedHallu's pairs do.
        question_id = f"{name}-{r['id']}"
        ex_id = f"{question_id}-{r['generation_model_name']}"
        out.append(Example(
            id=ex_id,
            dataset=name,
            split=_split_for(question_id),
            core=False,
            question=r["question"],
            documents=list(r["documents"]),
            response=r["response"],
            label=0 if r["adherence_score"] else 1,
            sentences=sentences,
            sentence_labels=[1 if k in unsupported else 0 for k in keys],
            meta={
                "pair": question_id,
                "generator": r["generation_model_name"],
                "annotator": r["annotating_model_name"],
                # free baselines RAGBench ships precomputed
                "ragas_faithfulness": r["ragas_faithfulness"],
                "trulens_groundedness": r["trulens_groundedness"],
            },
        ))
    return out


def _halubench(source: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/HaluBench", split="test", revision=REVISIONS["PatronusAI/HaluBench"])
    name = f"halubench-{source.lower().removesuffix('qa')}qa"
    out = []
    for r in ds:
        if r["source_ds"] != source:
            continue
        ex_id = f"{name}-{r['id']}"
        out.append(Example(
            id=ex_id,
            dataset=name,
            # row-level; _regroup_halubench() sets the question-level split
            split=_split_for(ex_id),
            core=False,
            question=r["question"],
            documents=[r["passage"]],
            response=r["answer"],
            label=1 if r["label"] == "FAIL" else 0,
            sentences=split_sentences(r["answer"]),
            meta={"pair": f"{name}-q-{_question_key(r['question'])}"},
        ))
    return out


def _question_key(question: str) -> str:
    return hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()[:16]


def _regroup_halubench(examples: list[Example]) -> None:
    """HaluBench has no question id, and its CovidQA subset asks most questions
    twice (a faithful and a perturbed answer), so its split is drawn per
    question, keyed on the question text. The PubMedQA subset has one row per
    question, so its row-id split already is a question split. The core slice
    is stratified on the row-level split, drawn just before this."""
    for ex in examples:
        if ex.dataset == "halubench-covidqa":
            ex.split = _split_for(ex.meta["pair"])


def _medhallu() -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", split="train",
                      revision=REVISIONS["UTAustin-AIHealth/MedHallu"])
    out = []
    for i, r in enumerate(ds):
        pair = f"medhallu-{i}"
        split = _split_for(pair)  # both answers of a question land in the same split
        knowledge = r["Knowledge"]
        documents = list(knowledge) if isinstance(knowledge, list) else [knowledge]
        meta = {
            "pair": pair,
            "difficulty": r["Difficulty Level"],
            "category": r["Category of Hallucination"],
        }
        for suffix, response, label in (("gt", r["Ground Truth"], 0), ("hal", r["Hallucinated Answer"], 1)):
            out.append(Example(
                id=f"{pair}-{suffix}",
                dataset="medhallu",
                split=split,
                core=False,
                question=r["Question"],
                documents=documents,
                response=response,
                label=label,
                sentences=split_sentences(response),
                meta=dict(meta),
            ))
    return out


_CITATION = re.compile(r"\s*\[\d+\]")


def _expertqa() -> list[Example]:
    if not EXPERTQA_PATH.exists():
        import httpx

        EXPERTQA_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPERTQA_PATH.write_bytes(httpx.get(EXPERTQA_URL, timeout=120, follow_redirects=True).content)
    out = []
    with EXPERTQA_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for q_idx, r in enumerate(rows):
        if r["metadata"]["field"] != "Healthcare / Medicine":
            continue
        group = f"expertqa-med-{q_idx}"
        for system in ("rr_gs_gpt4", "rr_sphere_gpt4"):
            answer = r["answers"].get(system)
            if not answer:
                continue
            for c_idx, claim in enumerate(answer.get("claims") or []):
                support = claim.get("support")
                evidence = [e for e in (claim.get("evidence") or []) if isinstance(e, str) and len(e) > 200]
                if support not in ("Complete", "Partial", "Incomplete") or not evidence:
                    continue
                # "[3] https://...\n\n passage" -> passage; the claim's own [n]
                # markers point at passage ids no detector can see
                documents = [e.split("\n\n", 1)[1].strip() if "\n\n" in e else e for e in evidence]
                text = _CITATION.sub("", claim["claim_string"]).strip()
                out.append(Example(
                    id=f"{group}-{system}-{c_idx}",
                    dataset="expertqa-med",
                    split=_split_for(group),
                    core=False,
                    question=r["question"],
                    documents=documents,
                    response=text,
                    label=0 if support == "Complete" else 1,
                    sentences=[text],
                    meta={
                        "pair": group,
                        "system": system,
                        "support": support,
                        "correctness": claim.get("correctness"),
                        "specific_field": r["metadata"].get("specific_field"),
                    },
                ))
    return out


def _mark_core(examples: list[Example], rng: random.Random) -> None:
    """Stratify by (dataset, split, label); MedHallu by question pair, so the
    core slice keeps its gold/hallucinated answers balanced the way the full
    set is."""
    strata = defaultdict(list)
    for ex in examples:
        if ex.dataset == "expertqa-med":
            # one entry per question, not per claim, so a question's claims are drawn together
            if ex.meta["pair"] not in strata[(ex.dataset, ex.split, "group")]:
                strata[(ex.dataset, ex.split, "group")].append(ex.meta["pair"])
        elif ex.dataset == "medhallu":
            if ex.id.endswith("-gt"):
                strata[(ex.dataset, ex.split, "pair")].append(ex.meta["pair"])
        else:
            strata[(ex.dataset, ex.split, ex.label)].append(ex.id)
    chosen: set[str] = set()
    for key in sorted(strata, key=str):
        members = sorted(strata[key])
        k = max(1, round(len(members) * CORE_SHARE))
        chosen.update(rng.sample(members, k))
    for ex in examples:
        ex.core = (ex.meta.get("pair") if ex.dataset in ("medhallu", "expertqa-med") else ex.id) in chosen


def build(path: Path = EXAMPLES_PATH) -> list[Example]:
    rng = random.Random(SEED)
    examples = (
        _ragbench("pubmedqa", RAGBENCH_PUBMEDQA_ROWS, rng)
        + _ragbench("covidqa", None, rng)
        + _halubench("pubmedQA")
        + _halubench("covidQA")
        + _medhallu()
    )
    _mark_core(examples, rng)
    _regroup_halubench(examples)
    # its own rng, so switching it on or off leaves every other dataset's
    # core slice the same
    if INCLUDE_EXPERTQA:
        expertqa = _expertqa()
        _mark_core(expertqa, random.Random(SEED + 1))
        examples += expertqa
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": Windows would otherwise write \r\n and change the sha256
    # printed by build-data, which is how machines check they hold the same rows
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
    return examples


def load_examples(
    path: Path = EXAMPLES_PATH,
    datasets: list[str] | None = None,
    core_only: bool = False,
    limit: int | None = None,
) -> list[Example]:
    if not path.exists():
        raise SystemExit(f"{path} missing - run `hallubench build-data` first")
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ex = Example(**json.loads(line))
            if datasets and ex.dataset not in datasets:
                continue
            if core_only and not ex.core:
                continue
            out.append(ex)
    if limit is not None:
        # spread a smoke-test limit over every dataset instead of the first one
        per = defaultdict(list)
        for ex in out:
            per[ex.dataset].append(ex)
        take = max(1, limit // max(1, len(per)))
        out = [ex for group in per.values() for ex in group[:take]]
    return out
