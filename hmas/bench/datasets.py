"""HumanEval+ / MBPP+ task loading (EvalPlus release files)."""
from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..config import ROOT

DATA_DIR = ROOT / "data" / "evalplus"  # mounted as the evalplus cache dir in the sandbox

# Pinned to the versions evalplus 0.3.1 expects, so its evaluator finds them in its cache.
DATASETS = {
    "humaneval": ("HumanEvalPlus", "v0.1.10"),
    "mbpp": ("MbppPlus", "v0.2.0"),
}


@dataclass
class Task:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str


def dataset_path(dataset: str) -> Path:
    name, version = DATASETS[dataset]
    return DATA_DIR / f"{name}-{version}.jsonl"


def download(dataset: str) -> Path:
    name, version = DATASETS[dataset]
    path = dataset_path(dataset)
    if path.exists():
        return path
    url = f"https://github.com/evalplus/{name.lower()}_release/releases/download/{version}/{name}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = gzip.decompress(r.read())
    path.write_bytes(data)
    return path


def load_tasks(dataset: str) -> list[Task]:
    path = dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/download_data.py")
    tasks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            tasks.append(Task(p["task_id"], p["prompt"], p["entry_point"], p["canonical_solution"]))
    return tasks
