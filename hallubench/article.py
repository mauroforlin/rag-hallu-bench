"""Every number the write-up shows, in one file: docs/data.json.

Built from results/report/metrics.json, results/report/run_stability.json
and the raw scores, so `hallubench evaluate && hallubench article` turns a
fresh set of runs into the article's figures with nothing copied by hand.
Besides what metrics.json already holds, it computes the "acting on a
confident score" figures on RAGBench alone, since the article reports them
both pooled and on the real answers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hallubench.data import PRIMARY_DATASETS, ROOT, load_examples
from hallubench.detectors import REGISTRY
from hallubench.metrics import acting_on_confidence
from hallubench.runner import RAW_DIR, load_results

DATA_PATH = ROOT / "docs" / "data.json"
REPORT = ROOT / "results" / "report"

# How each checker's number is produced, which is what the charts colour by
FAMILY = {
    "jev-choice": "calibrated", "jev-noul": "calibrated", "jev-sentences": "calibrated",
    "laya": "calibrated", "biodecision-v2-4b": "calibrated",
    "judge-logprob-qwen3-235b": "computed", "hhem": "computed", "minicheck-ft5": "computed",
    "minicheck-deberta": "computed", "alignscore-large": "computed", "lettucedetect-large": "computed",
    "bespoke-minicheck-7b": "computed", "granite-guardian-8b": "computed", "lynx-8b": "computed",
    "judge-qwen3-235b": "written", "judge-gpt-6-luna": "written", "judge-gpt-5.6-luna": "written", "judge-gemini-2.5-flash": "written",
    "judge-claude-haiku-5.5-thinking": "written", "judge-gemini-3.8-flash-thinking": "written",
    "ragas-faithfulness": "written", "trulens-groundedness": "written",
    "vertex-check-grounding": "service", "bedrock-contextual-grounding": "service",
    "baseline-length": "baseline", "baseline-novel-words": "baseline",
}
NAME = {
    "jev-choice": "Jev · Choice", "jev-noul": "Jev · Noul", "jev-sentences": "Jev · per sentence",
    "laya": "Laya", "biodecision-v2-4b": "BioDecision v2 4B",
    "judge-logprob-qwen3-235b": "Qwen3-235B · yes/no probability", "hhem": "HHEM-2.1",
    "minicheck-ft5": "MiniCheck Flan-T5", "minicheck-deberta": "MiniCheck DeBERTa",
    "alignscore-large": "AlignScore-large", "lettucedetect-large": "LettuceDetect",
    "bespoke-minicheck-7b": "Bespoke-MiniCheck 7B", "granite-guardian-8b": "Granite Guardian 3.3",
    "lynx-8b": "Lynx 8B", "judge-qwen3-235b": "Qwen3-235B · written confidence", "judge-gpt-6-luna": "GPT-6 Luna", "judge-gpt-5.6-luna": "GPT-5.6 Luna",
    "judge-gemini-2.5-flash": "Gemini 2.5 Flash", "judge-claude-haiku-5.5-thinking": "Claude Haiku 5.5 · thinking",
    "judge-gemini-3.8-flash-thinking": "Gemini 3.8 Flash · thinking", "ragas-faithfulness": "RAGAS",
    "trulens-groundedness": "TruLens", "vertex-check-grounding": "Google Check Grounding",
    "bedrock-contextual-grounding": "AWS Bedrock grounding check",
    "baseline-length": "Answer length", "baseline-novel-words": "Words not in the documents",
}
RAGBENCH = tuple(d for d in PRIMARY_DATASETS if d.startswith("ragbench"))
HALUBENCH = tuple(d for d in PRIMARY_DATASETS if d.startswith("halubench"))
# Latency measured one request at a time: over the network from Italy for the
# API checkers, on a Colab A100 for the open ones. The two cloud services were
# only timed inside the main parallel run (median of the full run).
GPU_LATENCY = {"hhem", "minicheck-ft5", "minicheck-deberta", "alignscore-large", "bespoke-minicheck-7b",
               "granite-guardian-8b", "lynx-8b"}
PARALLEL_LATENCY = ("vertex-check-grounding", "bedrock-contextual-grounding")
# Google's Check Grounding price per 1,000 checks (list price, not reported per call)
VERTEX_USD_PER_1K = 0.00075


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _weighted(per_ds, n_by_ds):
    """Mean of the per-dataset AUROCs weighted by each test split's size, for checkers that cover all five."""
    if not all(d in per_ds for d in n_by_ds):
        return None
    return float(sum(per_ds[d]["auroc"] * n for d, n in n_by_ds.items()) / sum(n_by_ds.values()))


def _acting(examples, results, datasets) -> dict | None:
    rows = [(e, results[e.id]["score"]) for e in examples
            if e.split == "test" and e.dataset in datasets and results.get(e.id, {}).get("score") is not None]
    if not rows:
        return None
    a = acting_on_confidence(np.array([e.label for e, _ in rows]), np.array([s for _, s in rows], dtype=float),
                             np.array([e.meta.get("pair", e.id) for e, _ in rows]))
    a["n"] = len(rows)
    return a


def _vertex_alternatives(examples, raw) -> dict:
    """Google returns no score when it finds nothing to check, and the study
    grades those answers as not flagged (score 0). The macro AUROC under the two
    other readings, dropping them or scoring them 0.5, shows how much rests on it."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for label, value in (("as_not_flagged", 0.0), ("as_half", 0.5), ("dropped", None)):
        per_ds = {}
        for ds in PRIMARY_DATASETS:
            ys, ss = [], []
            for e in examples:
                r = raw.get(e.id, {})
                if e.split != "test" or e.dataset != ds or r.get("score") is None:
                    continue
                unscored = r.get("raw", {}).get("scored_by_service") is False
                if unscored and value is None:
                    continue
                ys.append(e.label)
                ss.append(value if unscored else r["score"])
            per_ds[ds] = float(roc_auc_score(ys, ss))
        out[label] = {"macro": float(np.mean(list(per_ds.values()))), "per_ds": per_ds}
    return out


def build(path: Path = DATA_PATH) -> dict:
    metrics = json.loads((REPORT / "metrics.json").read_text(encoding="utf-8"))
    stability_path = REPORT / "run_stability.json"
    stability = json.loads(stability_path.read_text(encoding="utf-8")) if stability_path.exists() else {}
    examples = load_examples()
    paired = metrics["paired_vs_reference"]
    test = [e for e in examples if e.split == "test" and e.dataset in PRIMARY_DATASETS]
    n_by_ds = {d: sum(1 for e in test if e.dataset == d) for d in PRIMARY_DATASETS}
    dets = []
    for det_id, m in metrics["detectors"].items():
        if det_id not in FAMILY:
            continue  # prompt-ablation variants are reported separately
        per_ds = {ds: {"auroc": v["auroc"], "ci": v.get("auroc_ci")} for ds, v in m["by_dataset"].items()
                  if ds in PRIMARY_DATASETS and v.get("auroc") is not None}
        raw = load_results(det_id, RAW_DIR)
        entry = {
            "id": det_id, "name": NAME[det_id], "family": FAMILY[det_id],
            "macro": m["macro_auroc"],
            "ragbench": _mean([per_ds.get(d, {}).get("auroc") for d in RAGBENCH]),
            "halubench": _mean([per_ds.get(d, {}).get("auroc") for d in HALUBENCH]) if all(d in per_ds for d in HALUBENCH) else None,
            "per_ds": per_ds,
            # two other ways to average, to show the ranking doesn't rest on RAGBench CovidQA's few positives
            "weighted": _weighted(per_ds, n_by_ds),
            "macro_without_rb_covidqa": _mean([per_ds[d]["auroc"] for d in PRIMARY_DATASETS if d != "ragbench-covidqa"])
            if all(d in per_ds for d in PRIMARY_DATASETS) else None,
            "coverage": m["coverage"], "unscored": m["errors"],
            "cost_per_1k": m["cost"].get("usd_per_1k_rows"),
            "ece": m["core"].get("ece"),
            "medhallu_gold_flag": m.get("medhallu_gold_flag_rate"),
            "sentence_auroc": {ds: v["sentence_level"]["auroc"] for ds, v in m["by_dataset"].items()
                               if v.get("sentence_level")} or None,
        }
        if det_id == "vertex-check-grounding":
            entry["cost_per_1k"] = VERTEX_USD_PER_1K
            entry["no_checkable_claims"] = sum(1 for r in raw.values() if r.get("raw", {}).get("scored_by_service") is False)
            entry["unscored_alternatives"] = _vertex_alternatives(examples, raw)
        if FAMILY[det_id] not in ("baseline",) and REGISTRY[det_id].family != "precomputed":
            entry["acting"] = m.get("acting")
            if entry["acting"]:
                # rows behind the pooled figures, so the charts can show how many flags a share rests on
                entry["acting"]["n"] = sum(1 for e in examples if e.split == "test" and e.dataset in PRIMARY_DATASETS
                                           and raw.get(e.id, {}).get("score") is not None)
            entry["acting_ragbench"] = _acting(examples, raw, RAGBENCH)
        vs = paired.get(det_id)
        if vs:
            # Jev Choice minus this checker, per dataset; "ahead"/"behind" = Holm-adjusted p < 0.05
            entry["vs_jev"] = {ds: {"delta": v["delta"], "ci": v["ci"], "p_holm": v.get("p_holm")}
                               for ds, v in vs.items() if v and ds in PRIMARY_DATASETS}
            entry["jev_ahead"] = sum(1 for v in entry["vs_jev"].values() if v["p_holm"] is not None and v["p_holm"] < 0.05 and v["delta"] > 0)
            entry["jev_behind"] = sum(1 for v in entry["vs_jev"].values() if v["p_holm"] is not None and v["p_holm"] < 0.05 and v["delta"] < 0)
        dets.append(entry)

    latency = []
    for det_id, v in metrics.get("latency_sequential", {}).items():
        if det_id in FAMILY:
            latency.append({"id": det_id, "median": v["median_ms"], "p90": v["p90_ms"], "n": v["n"],
                            "where": "gpu" if det_id in GPU_LATENCY else "api"})
    for det_id in PARALLEL_LATENCY:
        lat = np.array([r["latency_ms"] for r in load_results(det_id, RAW_DIR).values() if r.get("latency_ms")])
        if len(lat):
            latency.append({"id": det_id, "median": float(np.median(lat)), "p90": float(np.percentile(lat, 90)),
                            "n": int(len(lat)), "where": "api-parallel"})

    data = {
        "detectors": dets,
        "ablation": [{"variant": k, **v} for k, v in metrics.get("prompt_ablation", {}).items()],
        "stability": [{"id": k, "macro1": v["macro_auroc_run1"], "macro2": v["macro_auroc_run2"],
                       "flips": v["verdict_flips_at_0.5"], "spearman": v["spearman"]} for k, v in stability.items()],
        "latency": latency,
        "n_tests_holm": metrics.get("n_tests_holm"),
        "n_test": sum(1 for e in examples if e.split == "test" and e.dataset in PRIMARY_DATASETS),
        "n_test_ragbench": sum(1 for e in examples if e.split == "test" and e.dataset in RAGBENCH),
        "test_by_ds": {d: {"n": n_by_ds[d], "hallucinated": sum(1 for e in test if e.dataset == d and e.label == 1)}
                       for d in PRIMARY_DATASETS},
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    return data
