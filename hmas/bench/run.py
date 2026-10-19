"""Generate solutions for a benchmark with a given agent.

Example:
    python -m hmas.bench.run --dataset humaneval --agent direct --model deepseek-chat
    python -m hmas.bench.run --dataset humaneval --agent direct --model deepseek-chat --limit 20

Writes results/<run_name>/ with:
    records.jsonl   one line per task: solution, usage, trace (resumable)
    samples.jsonl   EvalPlus input ({"task_id", "solution"})
    meta.json       run configuration
Then score it with: python -m hmas.bench.evaluate results/<run_name>
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import ROOT, load_model
from .datasets import load_tasks

RESULTS = ROOT / "results"


def build_agent(name: str, model: str | None):
    from ..agents.single import CanonicalAgent, DirectAgent

    if name == "canonical":
        return CanonicalAgent()
    if model is None:
        sys.exit("--model is required for LLM agents")
    from ..llm import LLMClient

    llm = LLMClient(load_model(model))
    if name == "direct":
        return DirectAgent(llm)
    sys.exit(f"unknown agent {name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--agent", default="direct")
    ap.add_argument("--model", default=None, help="name in configs/models.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (smoke tests)")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    tasks = load_tasks(args.dataset)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.task_id in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    run_name = args.run_name or f"{args.dataset}_{args.agent}_{args.model or 'none'}" + (
        f"_n{len(tasks)}" if args.limit or args.tasks else "")
    out = RESULTS / run_name
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"

    done = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["task_id"]] = r
    todo = [t for t in tasks if t.task_id not in done]
    print(f"[run] {run_name}: {len(tasks)} tasks, {len(done)} already done, {len(todo)} to go")

    agent = build_agent(args.agent, args.model)
    lock = threading.Lock()
    t_start = time.time()

    def work(task):
        t0 = time.time()
        try:
            res = agent.solve(task, args.dataset)
            rec = {"task_id": task.task_id, "solution": res.solution, "usage": res.usage.as_dict(),
                   "wall_time": round(time.time() - t0, 3), "trace": res.trace, "error": None}
        except Exception as e:  # keep going; failed tasks are retried on the next run
            rec = {"task_id": task.task_id, "error": f"{type(e).__name__}: {e}"}
        return rec

    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, rec_path.open("a", encoding="utf-8") as f:
        futs = [ex.submit(work, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec["error"] is None:
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                done[rec["task_id"]] = rec
            else:
                errors += 1
                print(f"[error] {rec['task_id']}: {rec['error']}", file=sys.stderr)
            if i % 20 == 0 or i == len(futs):
                print(f"[run] {i}/{len(futs)}  {time.time() - t_start:.0f}s")

    order = [t.task_id for t in tasks if t.task_id in done]
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for tid in order:
            f.write(json.dumps({"task_id": tid, "solution": done[tid]["solution"]}, ensure_ascii=False) + "\n")
    meta = {"dataset": args.dataset, "agent": args.agent, "model": args.model,
            "task_ids": [t.task_id for t in tasks], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.model:
        cfg = load_model(args.model)
        meta["model_config"] = {k: getattr(cfg, k) for k in
                                ("model", "base_url", "temperature", "max_tokens", "price_in", "price_out")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run] wrote {out}/samples.jsonl ({len(order)}/{len(tasks)} tasks, {errors} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
