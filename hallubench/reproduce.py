"""Rebuild every published number from the published scores, without calling
any model.

`hallubench pack` writes results/release/scores.zip: one line per answer for
every checker in the registry (main runs, the prompt-ablation variants, the
single-request latency samples and the second full run of Jev and Qwen). It is
attached to the GitHub release.

`hallubench reproduce` downloads that archive, rebuilds the rows from the
pinned datasets and checks their sha256, grades everything, regenerates the
article's numbers and compares them with docs/data.json as committed. A
mismatch means something in the pipeline differs from the published one.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

from hallubench import data
from hallubench.detectors import REGISTRY

RELEASE_URL = "https://github.com/mauroforlin/rag-hallu-bench/releases/download/v1.0/scores.zip"
EXPECTED_SHA = "df950765b366f89c"
RESULTS = data.ROOT / "results"
PACK_PATH = RESULTS / "release" / "scores.zip"
# folders inside the archive, and where each one goes
FOLDERS = {"raw": RESULTS / "raw", "latency": RESULTS / "latency", "rerun1": RESULTS / "rerun1"}
STABILITY = ["jev-choice", "jev-noul", "jev-sentences", "judge-logprob-qwen3-235b"]


def _clean(path: Path, ids: set[str]) -> str:
    """Last line per answer wins, as in runner.load_results; answers outside
    the study's rows are dropped."""
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in ids:
                rows[row["id"]] = json.dumps(row, ensure_ascii=False)
    return "".join(line + "\n" for line in rows.values())


def pack(path: Path = PACK_PATH) -> Path:
    ids = {e.id for e in data.load_examples()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, src in FOLDERS.items():
            for name in REGISTRY:
                f = src / f"{name}.jsonl"
                if f.exists():
                    z.writestr(f"{folder}/{name}.jsonl", _clean(f, ids))
    return path


def _sha() -> str:
    return hashlib.sha256(data.EXAMPLES_PATH.read_bytes()).hexdigest()[:16]


def _diff(a, b, where: str = "", out: list | None = None, tol: float = 1e-9) -> list:
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{where}.{k}: only in {'published' if k in a else 'rebuilt'}")
            else:
                _diff(a[k], b[k], f"{where}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: {len(a)} items published, {len(b)} rebuilt")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{where}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            out.append(f"{where}: published {a!r}, rebuilt {b!r}")
    elif a != b:
        out.append(f"{where}: published {a!r}, rebuilt {b!r}")
    return out


def reproduce(force: bool = False, archive_path: Path | None = None) -> bool:
    import httpx

    from hallubench import article
    from hallubench.metrics import compare_runs, evaluate, latency_summary, write

    if any(FOLDERS["raw"].glob("*.jsonl")) and not force:
        raise SystemExit("results/raw already holds scores; move them away or pass --force to overwrite them")
    if archive_path:
        archive = Path(archive_path).read_bytes()
    else:
        print(f"downloading {RELEASE_URL}")
        resp = httpx.get(RELEASE_URL, follow_redirects=True, timeout=300)
        resp.raise_for_status()
        archive = resp.content
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.namelist():
            folder, name = member.split("/", 1)
            dest = FOLDERS[folder] / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(member))
    print("building the rows from the pinned datasets")
    data.build()
    if _sha() != EXPECTED_SHA:
        raise SystemExit(f"rows differ from the published ones: sha256 {_sha()} instead of {EXPECTED_SHA}")
    examples = data.load_examples()
    print("grading (a few minutes: every interval is a 20,000-sample bootstrap)")
    report = evaluate(examples, raw_dir=FOLDERS["raw"])
    report["latency_sequential"] = latency_summary(FOLDERS["latency"])
    write(report, RESULTS / "report")
    stability = compare_runs(examples, FOLDERS["raw"], FOLDERS["rerun1"], STABILITY)
    (RESULTS / "report" / "run_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    published = json.loads(article.DATA_PATH.read_text(encoding="utf-8"))
    rebuilt_path = RESULTS / "report" / "data.rebuilt.json"
    article.build(rebuilt_path)
    rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))  # as written, so both sides went through JSON
    diffs = _diff(published, rebuilt)
    if diffs:
        print(f"{len(diffs)} numbers differ from docs/data.json, for example:")
        for d in diffs[:15]:
            print("  ", d)
        return False
    print("every number in docs/data.json was rebuilt identically from the published scores")
    return True
