"""hallubench build-data | list | run DETECTOR... | evaluate"""
from __future__ import annotations

import argparse
import json

from hallubench import data
from hallubench.detectors import REGISTRY


def _cmd_build(args) -> None:
    examples = data.build()
    by = {}
    for ex in examples:
        k = (ex.dataset, ex.split)
        by.setdefault(k, [0, 0, 0])
        by[k][0] += 1
        by[k][1] += ex.label
        by[k][2] += ex.core
    import hashlib
    digest = hashlib.sha256(data.EXAMPLES_PATH.read_bytes()).hexdigest()[:16]
    print(f"{len(examples)} rows -> {data.EXAMPLES_PATH}  (sha256 {digest}: must match on every machine)")
    for (ds, split), (n, pos, core) in sorted(by.items()):
        print(f"  {ds:20s} {split:4s} n={n:5d} hallucinated={pos / n:.0%} core={core}")


def _cmd_list(args) -> None:
    for name, spec in REGISTRY.items():
        extra = " (core slice only)" if spec.core_only else ""
        print(f"{name:28s} {spec.runs_on:4s} {spec.family}{extra}")


def _cmd_run(args) -> None:
    from hallubench.runner import run

    names = list(args.detectors)
    if args.where:
        names += [n for n, s in REGISTRY.items() if s.runs_on in args.where and n not in names]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown detector(s): {unknown}; see `hallubench list`")
    examples = data.load_examples(datasets=args.datasets, core_only=args.core, limit=args.limit)
    for name in names:
        counts = run(name, examples, only_errors=args.only_errors, workers=args.workers)
        print(f"{name}: {counts}")


def _cmd_evaluate(args) -> None:
    from hallubench.metrics import evaluate, write
    from hallubench.report import summary_table

    examples = data.load_examples()
    report = evaluate(examples, reference=args.reference)
    latency_dir = data.ROOT / "results" / "latency"
    if latency_dir.exists():
        from hallubench.metrics import latency_summary
        report["latency_sequential"] = latency_summary(latency_dir)
    path = write(report, data.ROOT / "results" / "report")
    print(summary_table(report))
    print(f"\nfull metrics: {path}")


def _cmd_compare_runs(args) -> None:
    from pathlib import Path

    from hallubench.metrics import compare_runs

    out = compare_runs(data.load_examples(), Path(args.run_a), Path(args.run_b), args.detectors)
    path = data.ROOT / "results" / "report" / "run_stability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in out.items():
        print(f"{name:28s} macro AUROC {v['macro_auroc_run1']:.3f} -> {v['macro_auroc_run2']:.3f}  "
              f"identical {v['identical_scores']:.0%}  verdict flips {v['verdict_flips_at_0.5']:.1%}  "
              f"spearman {v['spearman']:.3f}")
    print()
    print(path)


def _cmd_article(args) -> None:
    from hallubench.article import DATA_PATH, build

    data = build()
    print(f"{len(data['detectors'])} checkers -> {DATA_PATH}")


def _cmd_pack(args) -> None:
    from hallubench.reproduce import pack

    print(pack())


def _cmd_reproduce(args) -> None:
    from hallubench.reproduce import reproduce

    raise SystemExit(0 if reproduce(force=args.force, archive_path=args.archive) else 1)


def main() -> None:
    p = argparse.ArgumentParser(prog="hallubench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="download + sample the three datasets into data/examples.jsonl")
    sub.add_parser("list", help="list detectors")
    r = sub.add_parser("run", help="run detectors (resumable)")
    r.add_argument("detectors", nargs="*")
    r.add_argument("--where", nargs="+", choices=["api", "cpu", "gpu"],
                   help="also run every detector meant for these places")
    r.add_argument("--datasets", nargs="+", choices=data.DATASETS)
    r.add_argument("--core", action="store_true", help="only the stratified core slice")
    r.add_argument("--limit", type=int, help="smoke test: about this many rows, spread over datasets")
    r.add_argument("--workers", type=int, help="override concurrency (API) or batch size (local)")
    r.add_argument("--only-errors", action="store_true", help="retry only rows that previously failed")
    e = sub.add_parser("evaluate", help="grade everything in results/raw/")
    e.add_argument("--reference", default="jev-choice", help="detector the paired deltas compare against")
    c = sub.add_parser("compare-runs", help="run-to-run stability between two results folders")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("detectors", nargs="+")
    sub.add_parser("article", help="write docs/data.json, every number the write-up shows")
    sub.add_parser("pack", help="zip every checker's scores for the GitHub release")
    rp = sub.add_parser("reproduce", help="download the published scores and rebuild every number from them")
    rp.add_argument("--force", action="store_true", help="overwrite scores already in results/")
    rp.add_argument("--archive", help="use a local scores.zip instead of downloading the release")
    args = p.parse_args()
    {"build-data": _cmd_build, "list": _cmd_list, "run": _cmd_run, "evaluate": _cmd_evaluate,
     "compare-runs": _cmd_compare_runs, "article": _cmd_article,
     "pack": _cmd_pack, "reproduce": _cmd_reproduce}[args.cmd](args)


if __name__ == "__main__":
    main()
