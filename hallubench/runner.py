"""Run one detector over the rows, appending to results/raw/<detector>.jsonl.

Resumable: a row already in the file with a score is never re-run, so a
Colab session that dies at row 3000 picks up at 3001, and results from
different machines merge by just copying their files into results/raw/.
Rows that errored are retried on the next run; they stay in the file until
then and are counted as errors, not dropped, by the report.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from hallubench.data import ROOT, Example
from hallubench.detectors import REGISTRY
from hallubench.detectors.base import Detector, Result

# HALLUBENCH_RAW points smoke tests elsewhere, so they never mix with real runs
RAW_DIR = Path(os.environ.get("HALLUBENCH_RAW", ROOT / "results" / "raw"))


def results_path(name: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{name}.jsonl"


def load_results(name: str, raw_dir: Path = RAW_DIR) -> dict[str, dict]:
    """Last line per id wins, so a retried row replaces its earlier error."""
    path = results_path(name, raw_dir)
    out: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def _batched(det: Detector) -> bool:
    return type(det).score_batch is not Detector.score_batch


def run(
    name: str, examples: list[Example], raw_dir: Path = RAW_DIR, only_errors: bool = False,
    workers: int | None = None,
) -> dict[str, int]:
    spec = REGISTRY[name]
    if spec.datasets:
        examples = [ex for ex in examples if ex.dataset in spec.datasets]
    if spec.core_only:
        examples = [ex for ex in examples if ex.core]
    previous = load_results(name, raw_dir)
    done = {i for i, r in previous.items() if r.get("score") is not None}
    todo = [ex for ex in examples if ex.id not in done]
    if only_errors:  # retry failures without extending a deliberately partial run
        todo = [ex for ex in todo if ex.id in previous]
    counts = {"skipped": len(examples) - len(todo), "scored": 0, "errors": 0}
    if not todo:
        return counts

    det = spec.factory()
    if workers:
        det.workers = workers
    det.setup()
    path = results_path(name, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    with path.open("a", encoding="utf-8") as f, tqdm(total=len(todo), desc=name) as bar:
        def write(results: list[Result]) -> None:
            with lock:
                for r in results:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                    counts["errors" if r.score is None else "scored"] += 1
                f.flush()
                bar.update(len(results))

        def failed(batch: list[Example]) -> list[Result]:
            err = traceback.format_exc(limit=2).strip().splitlines()[-1]
            return [Result(id=ex.id, score=None, error=err) for ex in batch]

        if _batched(det):
            for i in range(0, len(todo), det.workers):
                batch = todo[i:i + det.workers]
                try:
                    write(det.score_batch(batch))
                except Exception:
                    write(failed(batch))
        else:
            def one(ex: Example) -> list[Result]:
                try:
                    return [det.score_one(ex)]
                except Exception:
                    return failed([ex])

            with ThreadPoolExecutor(max_workers=det.workers) as pool:
                for fut in as_completed([pool.submit(one, ex) for ex in todo]):
                    write(fut.result())
    return counts
