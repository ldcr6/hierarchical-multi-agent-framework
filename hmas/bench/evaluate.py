"""Score a run with the official EvalPlus evaluator, inside the Docker sandbox.

    python -m hmas.bench.evaluate results/<run_name>

Writes results/<run_name>/summary.json with pass@1 (base tests and plus tests),
LLM calls, tokens and cost. Subset runs (--limit) are scored on their own tasks:
EvalPlus insists on a sample for every task, so the others get an empty stub
that is excluded from the score.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from ..config import load_model
from ..sandbox import DockerSandbox
from .datasets import DATA_DIR, DATASETS, dataset_path, download, load_tasks


def summarize(run_dir: Path, eval_results: dict, meta: dict) -> dict:
    task_ids = meta["task_ids"]
    records = {}
    for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        records[r["task_id"]] = r

    per_task, base_ok, plus_ok = {}, 0, 0
    for tid in task_ids:
        res = eval_results["eval"].get(tid, [{}])[0]
        b = res.get("base_status") == "pass"
        p = b and res.get("plus_status") == "pass"
        base_ok += b
        plus_ok += p
        per_task[tid] = {"base": b, "plus": p}

    n = len(task_ids)
    calls = sum(records[t]["usage"]["calls"] for t in task_ids if t in records)
    pt = sum(records[t]["usage"]["prompt_tokens"] for t in task_ids if t in records)
    ct = sum(records[t]["usage"]["completion_tokens"] for t in task_ids if t in records)
    summary = {
        "run": run_dir.name, "dataset": meta["dataset"], "agent": meta["agent"], "model": meta["model"],
        "n_tasks": n, "n_generated": sum(t in records for t in task_ids),
        "pass@1_base": round(100 * base_ok / n, 2), "pass@1_plus": round(100 * plus_ok / n, 2),
        "llm_calls": calls, "avg_calls_per_task": round(calls / n, 3),
        "prompt_tokens": pt, "completion_tokens": ct, "avg_tokens_per_task": round((pt + ct) / n, 1),
        "evalplus_date": eval_results.get("date"), "evalplus_hash": eval_results.get("hash"),
    }
    if meta.get("model"):
        cost = load_model(meta["model"]).cost(pt, ct)
        summary["cost_cny"] = None if cost is None else round(cost, 4)
    summary["per_task"] = per_task
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    dataset = meta["dataset"]
    for ds in DATASETS:  # evalplus.sanitize loads both datasets
        download(ds)

    sb = DockerSandbox()
    if not sb.available():
        sys.exit("Docker is not running.")
    if not sb.image_exists():
        sys.exit(f"Sandbox image missing; build it: docker build -t {sb.image} -f docker/sandbox.Dockerfile docker")

    # full sample file: real solutions + empty stubs for tasks outside this run
    work = run_dir / "eval"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    samples = {}
    for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        samples[s["task_id"]] = s["solution"]
    with (work / "samples.jsonl").open("w", encoding="utf-8") as f:
        for t in load_tasks(dataset):
            f.write(json.dumps({"task_id": t.task_id, "solution": samples.get(t.task_id, "")}) + "\n")

    uid = f"{os.getuid()}:{os.getgid()}"
    cmd = ["sh", "-c",
           "evalplus.sanitize --samples /work/samples.jsonl && "
           f"evalplus.evaluate --dataset {dataset} --samples /work/samples-sanitized.jsonl "
           f"--parallel {args.parallel}"]
    print(f"[eval] running EvalPlus ({dataset}) in sandbox ...")
    r = sb.run_cmd(
        cmd,
        mounts=[(str(work), "/work", "rw"), (str(DATA_DIR), "/cache/evalplus", "rw")],
        env={"XDG_CACHE_HOME": "/cache", "HOME": "/tmp"},
        timeout=args.timeout, cpus=float(os.cpu_count() or 2), memory="8g", user=uid,
    )
    (work / "evalplus.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    res_path = work / "samples-sanitized_eval_results.json"
    if not r.ok or not res_path.exists():
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        sys.exit(f"[eval] EvalPlus failed (exit {r.returncode}, timed_out={r.timed_out}); log: {work}/evalplus.log")

    summary = summarize(run_dir, json.loads(res_path.read_text()), meta)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["n_tasks", "pass@1_base", "pass@1_plus", "avg_calls_per_task", "avg_tokens_per_task", "cost_cny"]
    print(f"[eval] {summary['run']}: " + ", ".join(f"{k}={summary.get(k)}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
