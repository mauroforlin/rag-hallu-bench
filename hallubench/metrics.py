"""Grade every detector's raw scores against the gold labels.

All headline numbers are on the *test* split. The dev split is used for one
thing only: choosing each detector's F1-optimal threshold, so the
thresholded numbers (precision, recall, F1) are what a team would get after
tuning on their own held-out data, not after peeking at the test set.

AUROC is the headline because it is threshold-free and indifferent to the
score's scale, which differs by detector (a probability for Jev, a fraction of
supported statements for RAGAS). Calibration (ECE) and the fixed-threshold
"confident flag" precision are reported only for detectors whose score is
meant to be read as a probability.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from hallubench.data import DATASETS, PRIMARY_DATASETS, Example
from hallubench.detectors import REGISTRY
from hallubench.runner import RAW_DIR, load_results

# 20,000 resamples: the smallest p a bootstrap can give is about 2/BOOTSTRAP, and
# Holm multiplies it by the number of tests (101 here), so fewer resamples would
# put a floor under every adjusted p near the 0.05 line
BOOTSTRAP = 20000
SEED = 0  # every bootstrap starts from it, so a CI doesn't depend on what else was graded first
CONFIDENT = 0.9  # "the detector is sure this is a hallucination"


def _auroc(y, s) -> float | None:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def _group_weights(groups: np.ndarray) -> np.ndarray:
    """BOOTSTRAP x n matrix: how many times each row is drawn when whole
    groups (a question's answers) are resampled with replacement. Drawing
    groups and concatenating their rows, the textbook cluster bootstrap, is
    the same distribution; weights just let every resample be scored at once."""
    keys, row_group = np.unique(groups, return_inverse=True)
    counts = np.random.default_rng(SEED).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=BOOTSTRAP)
    return counts[:, row_group].astype(float)


def _weighted_auroc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC of every resample (row of *w*) in one pass. Equals sklearn's
    roc_auc_score on the expanded sample, ties counted as half, as there.
    Resamples missing a class come back NaN and are dropped by the caller."""
    uniq, inv = np.unique(s, return_inverse=True)
    onehot = np.zeros((len(s), len(uniq)))
    onehot[np.arange(len(s)), inv] = 1.0
    pos = (w * (y == 1)) @ onehot  # resamples x unique scores
    neg = (w * (y == 0)) @ onehot
    below = np.cumsum(neg, axis=1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def _bootstrap_auroc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """95% CI, resampling whole groups (MedHallu's gold/hallucinated pair,
    RAGBench's two answers to a question) so related rows stay together."""
    vals = _weighted_auroc(y, s, _group_weights(groups))
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _best_threshold(y, s) -> float:
    """F1-optimal cut among 197 quantiles of the scores (all at once)."""
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    pred = s[None, :] >= cands[:, None]
    tp = (pred & (y == 1)).sum(1)
    f1 = 2 * tp / np.maximum(1, pred.sum(1) + (y == 1).sum())
    return float(cands[int(np.argmax(f1))])


def _ece(y, s, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) | (hi == 1.0))
        if m.any():
            total += m.sum() / len(s) * abs(y[m].mean() - s[m].mean())
    return float(total)


def reliability(y, s, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) | (hi == 1.0))
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "mean_score": float(s[m].mean()), "hallucination_rate": float(y[m].mean())})
    return out


def _grade(rows: list[tuple[Example, dict]], probabilistic: bool) -> dict:
    dev = [(e, r) for e, r in rows if e.split == "dev"]
    test = [(e, r) for e, r in rows if e.split == "test"]
    if not test:
        return {}
    y = np.array([e.label for e, _ in test])
    s = np.array([r["score"] for _, r in test], dtype=float)
    groups = np.array([e.meta.get("pair", e.id) for e, _ in test])
    out: dict = {"n_test": len(test), "base_rate": float(y.mean()), "auroc": _auroc(y, s)}
    if out["auroc"] is not None:
        out["auroc_ci"] = _bootstrap_auroc(y, s, groups)
        out["auprc"] = float(average_precision_score(y, s))
    if dev and len({e.label for e, _ in dev}) == 2:
        yd = np.array([e.label for e, _ in dev])
        sd = np.array([r["score"] for _, r in dev], dtype=float)
        t = _best_threshold(yd, sd)
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        out.update(
            threshold=t,
            precision=tp / max(1, int(pred.sum())),
            recall=tp / max(1, int(y.sum())),
            f1=float(f1_score(y, pred, zero_division=0)),
        )
    if probabilistic:
        flagged = s >= CONFIDENT
        cleared = s <= 1 - CONFIDENT
        out.update(
            ece=_ece(y, s),
            confident_flag_share=float(flagged.mean()),
            confident_flag_precision=float(y[flagged].mean()) if flagged.any() else None,
            confident_clear_share=float(cleared.mean()),
            confident_clear_npv=float(1 - y[cleared].mean()) if cleared.any() else None,
        )
    return out


def _global_threshold(rows: list[tuple[Example, dict]]) -> dict:
    """One threshold for everything, the way a team would deploy it: chosen
    for F1 on all datasets' dev rows pooled, then applied unchanged to each
    dataset's test rows. Shows whether a detector's scores mean the same thing
    across sources, which per-dataset tuning hides."""
    dev = [(e, r) for e, r in rows if e.split == "dev"]
    t = _best_threshold(np.array([e.label for e, _ in dev]), np.array([r["score"] for _, r in dev], dtype=float))
    out: dict = {"threshold": t, "by_dataset": {}}
    for ds in DATASETS:
        test = [(e, r) for e, r in rows if e.split == "test" and e.dataset == ds]
        if not test:
            continue
        y = np.array([e.label for e, _ in test])
        pred = np.array([r["score"] for _, r in test], dtype=float) >= t
        tp = int((pred & (y == 1)).sum())
        out["by_dataset"][ds] = {
            "precision": tp / max(1, int(pred.sum())),
            "recall": tp / max(1, int(y.sum())),
            "flag_rate": float(pred.mean()),
        }
    return out


def _sentence_auroc(rows: list[tuple[Example, dict]]) -> dict | None:
    ys, ss = [], []
    for e, r in rows:
        if e.split != "test" or e.sentence_labels is None or not r.get("sentence_scores"):
            continue
        if len(r["sentence_scores"]) != len(e.sentence_labels):
            continue
        ys.extend(e.sentence_labels)
        ss.extend(r["sentence_scores"])
    if len(set(ys)) < 2:
        return None
    return {"n_sentences": len(ys), "base_rate": float(np.mean(ys)), "auroc": _auroc(ys, ss)}


def _cost(rows: list[tuple[Example, dict]]) -> dict:
    lat = [r["latency_ms"] for _, r in rows if r.get("latency_ms") is not None]
    cost = [r["cost_usd"] for _, r in rows if r.get("cost_usd") is not None]
    return {
        "median_latency_ms": float(np.median(lat)) if lat else None,
        "p90_latency_ms": float(np.percentile(lat, 90)) if lat else None,
        "usd_total": float(sum(cost)) if cost else None,
        "usd_per_1k_rows": float(1000 * sum(cost) / len(cost)) if cost else None,
    }


def paired_delta(a: dict[str, dict], b: dict[str, dict], examples: list[Example]) -> dict | None:
    """AUROC(a) - AUROC(b) on the test rows both scored, with a paired
    bootstrap CI."""
    common = [e for e in examples if e.split == "test"
              and a.get(e.id, {}).get("score") is not None and b.get(e.id, {}).get("score") is not None]
    y = np.array([e.label for e in common])
    if len(set(y)) < 2:
        return None
    sa = np.array([a[e.id]["score"] for e in common], dtype=float)
    sb = np.array([b[e.id]["score"] for e in common], dtype=float)
    groups = np.array([e.meta.get("pair", e.id) for e in common])
    w = _group_weights(groups)  # the same resamples for both detectors: that's what makes it paired
    deltas = _weighted_auroc(y, sa, w) - _weighted_auroc(y, sb, w)
    deltas = deltas[~np.isnan(deltas)]
    # two-sided bootstrap p-value for "no difference", with the +1 that keeps
    # it from ever reading as exactly 0 on a finite number of resamples
    tail = min((deltas <= 0).sum(), (deltas >= 0).sum())
    return {
        "n": len(common),
        "delta": float(roc_auc_score(y, sa) - roc_auc_score(y, sb)),
        "ci": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "p": float(min(1.0, 2 * (tail + 1) / (len(deltas) + 1))),
    }


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values over one family of tests: with about
    100 comparisons, about five would cross p < 0.05 by chance alone."""
    keys = sorted(pvalues, key=pvalues.get)
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


PREVALENCES = (0.05, 0.10, 0.20)


def acting_on_confidence(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> dict:
    """What "≥ 90% sure it's hallucinated" and "≤ 10%" are worth at the
    hallucination rates a deployed system might see, besides this
    benchmark's ~40%. Precision depends on prevalence; the detector's own
    hit rates (TPR, FPR) don't, so they carry over and Bayes' rule does the
    rest: precision = π·TPR / (π·TPR + (1-π)·FPR). 95% CIs from the same
    cluster bootstrap as AUROC."""
    w = _group_weights(groups)
    pos, neg = w * (y == 1), w * (y == 0)
    flag, clear = s >= CONFIDENT, s <= 1 - CONFIDENT
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = (pos * flag).sum(1) / pos.sum(1)  # hallucinations flagged
        fpr = (neg * flag).sum(1) / neg.sum(1)  # faithful answers wrongly flagged
        miss = (pos * clear).sum(1) / pos.sum(1)  # hallucinations wrongly cleared
        spec = (neg * clear).sum(1) / neg.sum(1)  # faithful answers cleared

    def summarise(point: float, boot: np.ndarray) -> dict:
        boot = boot[~np.isnan(boot)]
        return {"value": point, "ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))}

    p_tpr, p_fpr = float(flag[y == 1].mean()), float(flag[y == 0].mean())
    p_miss, p_spec = float(clear[y == 1].mean()), float(clear[y == 0].mean())
    out: dict = {
        "observed_base_rate": float(y.mean()),
        "flag_share": float(flag.mean()), "flag_tpr": p_tpr, "flag_fpr": p_fpr,
        "clear_share": float(clear.mean()), "clear_miss_rate": p_miss, "clear_specificity": p_spec,
        "flag_precision": {}, "clear_npv": {},
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        for pi in PREVALENCES:
            def prec(t, f): return pi * t / (pi * t + (1 - pi) * f)
            def npv(m_, sp): return (1 - pi) * sp / ((1 - pi) * sp + pi * m_)
            out["flag_precision"][str(pi)] = summarise(float(prec(p_tpr, p_fpr)), prec(tpr, fpr))
            out["clear_npv"][str(pi)] = summarise(float(npv(p_miss, p_spec)), npv(miss, spec))
    return out


# Which subsets the ranking is re-run on, to see if a conclusion rests on one dataset
SENSITIVITY = {
    "all_five": PRIMARY_DATASETS,
    "without_medhallu": tuple(d for d in PRIMARY_DATASETS if d != "medhallu"),
    "without_halubench": tuple(d for d in PRIMARY_DATASETS if not d.startswith("halubench")),
    "ragbench_only": tuple(d for d in PRIMARY_DATASETS if d.startswith("ragbench")),
}

# Each prompt-sensitivity variant, paired with the detector it rewords
ABLATION_BASE = {
    "jev-choice-plain": "jev-choice",
    "jev-noul-flipped": "jev-noul",
    "judge-logprob-qwen3-235b-plain": "judge-logprob-qwen3-235b",
    "judge-logprob-qwen3-235b-flipped": "judge-logprob-qwen3-235b",
}


def _orient_on_dev(scored: list[tuple[Example, dict]]) -> list[tuple[Example, dict]]:
    """A surface feature can point either way: MedHallu's hallucinated answers
    are shorter than its gold ones, RAGBench's are longer. The trivial
    baselines get the direction their dev rows favour, per dataset, the way
    every other detector gets its threshold from dev, so each is the best the
    feature can do with honest tuning."""
    dev = defaultdict(list)
    for e, r in scored:
        if e.split == "dev":
            dev[e.dataset].append((e.label, r["score"]))
    flip = {ds: (_auroc(np.array([y for y, _ in rows]), np.array([s for _, s in rows])) or 0.5) < 0.5
            for ds, rows in dev.items()}
    return [(e, {**r, "score": 1.0 - r["score"]} if flip.get(e.dataset) else r) for e, r in scored]


def evaluate(examples: list[Example], raw_dir: Path = RAW_DIR, reference: str = "jev-choice") -> dict:
    by_id = {e.id: e for e in examples}
    report: dict = {"detectors": {}, "paired_vs_reference": {}, "reference": reference}
    all_results = {name: load_results(name, raw_dir) for name in REGISTRY}
    all_results = {k: v for k, v in all_results.items() if v}

    for name, results in all_results.items():
        spec = REGISTRY[name]
        # RAGAS/TruLens and the trivial baselines: a fraction, not a probability
        det_prob = spec.family not in ("precomputed", "baseline")
        eligible = [e for e in examples
                    if (not spec.datasets or e.dataset in spec.datasets) and (not spec.core_only or e.core)]
        scored = [(by_id[i], r) for i, r in results.items() if i in by_id and r.get("score") is not None]
        entry: dict = {
            "family": spec.family,
            "runs_on": spec.runs_on,
            "coverage": len(scored) / max(1, len(eligible)),
            "errors": sum(1 for i, r in results.items() if i in by_id and r.get("score") is None),
            "cost": _cost(scored),
            "latency_kind": spec.latency,
            "by_dataset": {},
            "core": _grade([(e, r) for e, r in scored if e.core], det_prob),
        }
        if spec.family == "baseline":
            scored = _orient_on_dev(scored)
        per_ds = defaultdict(list)
        for e, r in scored:
            per_ds[e.dataset].append((e, r))
        for ds in DATASETS:
            if per_ds.get(ds):
                entry["by_dataset"][ds] = _grade(per_ds[ds], det_prob)
                sent = _sentence_auroc(per_ds[ds])
                if sent:
                    entry["by_dataset"][ds]["sentence_level"] = sent
        aurocs = [entry["by_dataset"].get(ds, {}).get("auroc") for ds in PRIMARY_DATASETS]
        entry["macro_auroc"] = float(np.mean(aurocs)) if all(a is not None for a in aurocs) else None
        entry["macro_variants"] = {}
        for label, subset in SENSITIVITY.items():
            vals = [entry["by_dataset"].get(ds, {}).get("auroc") for ds in subset]
            entry["macro_variants"][label] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        pooled = [(e, r) for e, r in scored if e.split == "test" and e.dataset in PRIMARY_DATASETS]
        if det_prob and entry["macro_auroc"] is not None:
            entry["acting"] = acting_on_confidence(
                np.array([e.label for e, _ in pooled]),
                np.array([r["score"] for _, r in pooled], dtype=float),
                np.array([e.meta.get("pair", e.id) for e, _ in pooled]),
            )
        mh = per_ds.get("medhallu", [])
        if mh:
            entry["medhallu_by_difficulty"] = {
                d: _grade([(e, r) for e, r in mh if e.meta.get("difficulty") == d], det_prob)
                for d in ("easy", "medium", "hard")
            }
        if det_prob and entry["macro_auroc"] is not None:
            entry["global_threshold"] = _global_threshold(scored)
        mh_gold = [r["score"] for e, r in mh if e.split == "test" and e.label == 0]
        if det_prob and mh_gold:
            # MedHallu's gold answers can state conclusions their passages don't
            # (see data.py); this is how often each detector calls them ungrounded
            entry["medhallu_gold_flag_rate"] = float(np.mean(np.array(mh_gold) >= 0.5))
        core_test = [(e, r) for e, r in scored if e.core and e.split == "test"]
        if det_prob and core_test:
            entry["reliability_core"] = reliability(
                np.array([e.label for e, _ in core_test]),
                np.array([r["score"] for _, r in core_test], dtype=float),
            )
        report["detectors"][name] = entry

    ref = all_results.get(reference)
    if ref:
        for name, results in all_results.items():
            # ablation variants have their own section; the trivial baselines are
            # controls, graded on dev-oriented scores, not contenders
            if name == reference or REGISTRY[name].family in ("ablation", "baseline"):
                continue
            report["paired_vs_reference"][name] = {
                ds: paired_delta(ref, results, [e for e in examples if e.dataset == ds]) for ds in DATASETS
            }
        # one family of tests: every detector against the reference, on every primary dataset
        family = {(n, ds): v["p"] for n, row in report["paired_vs_reference"].items()
                  for ds, v in row.items() if v and ds in PRIMARY_DATASETS}
        for (n, ds), p_adj in holm(family).items():
            report["paired_vs_reference"][n][ds]["p_holm"] = p_adj
        report["n_tests_holm"] = len(family)

    core = [e for e in examples if e.core]
    report["prompt_ablation"] = {}
    for variant, base in ABLATION_BASE.items():
        if variant in all_results and base in all_results:
            d = paired_delta(all_results[variant], all_results[base], core)
            common = [e.id for e in core if e.split == "test"
                      and all_results[variant].get(e.id, {}).get("score") is not None
                      and all_results[base].get(e.id, {}).get("score") is not None]
            a = np.array([all_results[variant][i]["score"] for i in common])
            b = np.array([all_results[base][i]["score"] for i in common])
            report["prompt_ablation"][variant] = {
                "base": base, **(d or {}),
                "spearman": float(spearmanr(a, b).statistic),
                "verdict_agreement_at_0.5": float(((a >= 0.5) == (b >= 0.5)).mean()),
            }
    return report


def compare_runs(examples: list[Example], dir_a: Path, dir_b: Path, names: list[str]) -> dict:
    """Run-to-run stability: the same detector, same rows, asked twice."""

    out = {}
    for name in names:
        a, b = load_results(name, dir_a), load_results(name, dir_b)
        rows = [e for e in examples if a.get(e.id, {}).get("score") is not None
                and b.get(e.id, {}).get("score") is not None]
        if not rows:
            continue
        sa = np.array([a[e.id]["score"] for e in rows], dtype=float)
        sb = np.array([b[e.id]["score"] for e in rows], dtype=float)
        per_ds = {}
        for ds in PRIMARY_DATASETS:
            t = [i for i, e in enumerate(rows) if e.dataset == ds and e.split == "test"]
            y = np.array([rows[i].label for i in t])
            if len(set(y)) == 2:
                per_ds[ds] = {"auroc_run1": float(roc_auc_score(y, sa[t])), "auroc_run2": float(roc_auc_score(y, sb[t]))}
        macro = [np.mean([v[k] for v in per_ds.values()]) for k in ("auroc_run1", "auroc_run2")]
        out[name] = {
            "n": len(rows),
            "identical_scores": float((np.abs(sa - sb) < 1e-9).mean()),
            "mean_abs_score_change": float(np.abs(sa - sb).mean()),
            "spearman": float(spearmanr(sa, sb).statistic),
            "verdict_flips_at_0.5": float(((sa >= 0.5) != (sb >= 0.5)).mean()),
            "confident_flag_flips_at_0.9": float(((sa >= CONFIDENT) != (sb >= CONFIDENT)).mean()),
            "macro_auroc_run1": float(macro[0]), "macro_auroc_run2": float(macro[1]),
            "by_dataset": per_ds,
        }
    return out


def write(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def latency_summary(latency_dir: Path) -> dict:
    """Per-check wait measured one request at a time (`run --workers 1` into
    its own folder), the way one user waits in a RAG app. Kept apart from the
    main runs, which used many parallel requests and so measure a loaded service."""
    out = {}
    for path in sorted(latency_dir.glob("*.jsonl")):
        lat = [json.loads(line).get("latency_ms") for line in path.open(encoding="utf-8") if line.strip()]
        lat = np.array([x for x in lat if x is not None], dtype=float)
        if len(lat):
            out[path.stem] = {"n": int(len(lat)), "median_ms": float(np.median(lat)),
                              "p90_ms": float(np.percentile(lat, 90)), "mean_ms": float(lat.mean())}
    return out
