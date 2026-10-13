"""Plain-text summary of metrics.json, for the terminal and the README.

Ranked by macro AUROC: the mean of the test AUROCs on the five primary
datasets (ExpertQA-Med is shown beside them, see data.PRIMARY_DATASETS). Pooling
the datasets into one AUROC would reward a detector for scoring whole
datasets differently (HaluBench answers are short, RAGBench ones long), which
is not the skill being measured. A detector that didn't cover every dataset
has no macro AUROC and is listed after the others with its per-dataset numbers.
"""
from __future__ import annotations

from hallubench.data import DATASETS

_SHORT = {
    "ragbench-pubmedqa": "RB-PubMed",
    "ragbench-covidqa": "RB-Covid",
    "halubench-pubmedqa": "HB-PubMed",
    "halubench-covidqa": "HB-Covid",
    "medhallu": "MedHallu",
    "expertqa-med": "ExpertQA-Med",
}


def _f(x, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def summary_table(report: dict) -> str:
    dets = report["detectors"]
    dets = {n: d for n, d in dets.items() if d["family"] != "ablation"}  # reported in their own section
    order = sorted(dets, key=lambda n: (dets[n]["macro_auroc"] is None, -(dets[n]["macro_auroc"] or 0)))
    head = ["detector", "macro AUROC", *(_SHORT[d] for d in DATASETS), "coverage", "ECE (core)",
            "flag>=0.9 precision (core)", "$/1k answers"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in order:
        d = dets[n]
        cells = [
            n,
            _f(d["macro_auroc"]),
            *(_f(d["by_dataset"].get(ds, {}).get("auroc")) for ds in DATASETS),
            # one decimal below 100%, so a few unscored rows don't round away
            f"{d['coverage']:.0%}" if d["coverage"] >= 1 else f"{d['coverage']:.1%}",
            _f(d["core"].get("ece")),
            _f(d["core"].get("confident_flag_precision")),
            _f(d["cost"].get("usd_per_1k_rows"), 2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
