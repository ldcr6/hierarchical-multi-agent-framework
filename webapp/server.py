"""FastAPI backend of the demo system.

Wraps the Planner -> Coder -> Verifier framework (hmas.framework, v0) as an HTTP API:
submit a problem, then poll the run to watch the decomposition, each code version,
the sandbox test results and the final program. The framework code is used as is;
this module only builds the inputs, runs the graph in a worker thread and turns
the in-progress state into JSON.

Start (from the repository root):
    pip install -e ".[web]"
    python -m webapp.server            # http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hmas.bench.datasets import Task
from hmas.config import ROOT, load_dotenv, load_model
from hmas.framework.agents import Run
from hmas.framework.config import PRESETS
from hmas.framework.graph import HierarchicalAgent
from hmas.framework.harness import run_tests
from hmas.framework.program import parse_assert_block
from hmas.sandbox import DockerSandbox

from .examples import BY_ID, ScriptedLLM, public_examples
from .local_sandbox import LocalSandbox

STATIC = Path(__file__).resolve().parent / "static"
HISTORY = ROOT / "results" / "webapp_runs"   # results/ is the place for run outputs; this folder is git-ignored
LIVE_MODELS = ("deepseek-flash", "deepseek-flash-thinking", "qwen-coder-plus", "qwen2.5-coder-7b-local")

app = FastAPI(title="HMAS 演示系统", version="0.1.0")
_runs: dict[str, dict] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------- environment

def _docker() -> DockerSandbox | None:
    sb = DockerSandbox(timeout=60)
    try:
        return sb if sb.available() and sb.image_exists() else None
    except OSError:
        return None


def _sandbox_for(mode: str):
    """Real model output only ever runs in Docker; scripted (our own) code may use the local runner."""
    sb = _docker()
    if sb is not None:
        return sb, "docker"
    if mode == "offline" and LocalSandbox.available():
        return LocalSandbox(), "local"
    if mode == "offline":
        raise HTTPException(400, "离线演示也需要 Docker 沙箱（Windows 没有本地替代）：请先启动 Docker Desktop 并构建 hmas-sandbox:0.2")
    raise HTTPException(400, "真实模型模式必须在 Docker 沙箱里运行生成的代码：请先启动 Docker Desktop 并构建 hmas-sandbox:0.2")


def _model_status() -> list[dict]:
    load_dotenv()
    out = []
    for name in LIVE_MODELS:
        try:
            cfg = load_model(name)
        except Exception:
            continue
        try:
            has_key = bool(cfg.api_key)
        except RuntimeError:
            has_key = False
        out.append({"name": name, "model": cfg.model, "base_url": cfg.base_url, "has_key": has_key,
                    "key_env": cfg.api_key_env})
    return out


# ---------------------------------------------------------------- problem input

class RunRequest(BaseModel):
    mode: str = Field("offline", pattern="^(offline|live)$")
    example_id: str | None = None
    style: str = Field("humaneval", pattern="^(humaneval|mbpp)$")
    prompt: str = ""
    entry_point: str | None = None
    acceptance: str = ""             # optional extra asserts, run on the final program only
    model: str = "deepseek-flash"
    confirm_paid: bool = False


def _entry_point_from_stub(prompt: str) -> str:
    try:
        tree = ast.parse(prompt.rstrip() + "\n    pass\n")
    except SyntaxError as e:
        raise HTTPException(400, f"函数签名 + 文档字符串无法解析：{e.msg}（第 {e.lineno} 行）")
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise HTTPException(400, "没有找到函数定义：请写成 def 函数名(参数): 加文档字符串")
    return funcs[-1]   # HumanEval convention: the last function is the entry point


def _build_task(req: RunRequest) -> tuple[Task, str]:
    if req.example_id:
        ex = BY_ID.get(req.example_id)
        if ex is None:
            raise HTTPException(404, f"没有示例 {req.example_id}")
        prompt, style, entry = ex["prompt"], ex["style"], ex["entry_point"]
    else:
        prompt, style = req.prompt, req.style
        if not prompt.strip():
            raise HTTPException(400, "题目不能为空")
        if style == "humaneval":
            entry = _entry_point_from_stub(prompt)
        else:
            entry = (req.entry_point or "").strip()
            if not entry.isidentifier():
                raise HTTPException(400, "请填写入口函数名（合法的 Python 标识符）")
    if style == "humaneval" and not prompt.endswith("\n"):
        prompt += "\n"
    if style == "mbpp":
        prompt = '"""\n' + prompt.strip() + '\n"""\n'
    return Task(task_id=f"web/{entry}", prompt=prompt, entry_point=entry, canonical_solution=""), style


# ---------------------------------------------------------------- state -> JSON

def _cand_json(c, n_visible_default: int) -> dict:
    r = c.report
    results = []
    if r is not None:
        nv = c.n_visible
        for i, x in enumerate(r.results):
            results.append({"test": x["test"], "ok": x["ok"], "error": x["error"],
                            "kind": "visible" if i < nv else "generated"})
    return {
        "origin": c.origin, "code": c.code, "static_error": c.static_error, "passed": c.passed,
        "score": list(c.score()),
        "load_error": r.load_error if r else None, "sandbox_error": r.sandbox_error if r else None,
        "tests_passed": r.passed if r else None, "tests_total": r.total if r else None,
        "results": results,
    }


def _node_json(n, current: bool) -> dict:
    cands = list(n.candidates)
    if n.final is not None:
        status = "passed" if n.final.passed else "failed"
    elif current:
        status = "running"
    else:
        status = "pending"
    return {
        "name": n.name, "root": n.is_root, "signature": n.signature, "description": n.description,
        "depends_on": list(n.depends_on), "complexity": n.complexity, "confidence": n.confidence,
        "level": n.level, "visible_tests": list(n.visible), "generated_tests": list(n.tests or []),
        "candidates": [_cand_json(c, len(n.visible)) for c in cands],
        "final_index": cands.index(n.final) if n.final in cands else None,
        "status": status,
    }


def _snapshot(job: dict) -> dict:
    run: Run | None = job.get("run")
    out = {k: job[k] for k in ("id", "status", "mode", "model", "title", "style", "entry_point", "prompt",
                                "created", "error", "sandbox", "acceptance_results", "final_code")}
    out["elapsed"] = round((job.get("finished") or time.time()) - job["started"], 1) if job.get("started") else 0
    if run is None:
        out.update(plan=None, nodes=[], trace=[], usage=None)
        return out
    nodes = list(run.nodes)
    cursor = next((i for i, n in enumerate(nodes) if n.final is None), None)
    out["plan"] = None if run.root is None else {
        "root_complexity": run.root.complexity, "composition": run.composition,
        "problems": list(run.plan_problems), "n_helpers": len(run.helpers),
    }
    out["nodes"] = [_node_json(n, job["status"] == "running" and i == cursor) for i, n in enumerate(nodes)]
    out["trace"] = [dict(t) for t in list(run.trace)]
    usage = {r: u.as_dict() for r, u in run.usage.items()}
    tot = run.total_usage().as_dict()
    out["usage"] = {"by_role": usage, "total": tot}
    return out


# ---------------------------------------------------------------- worker

def _execute(job_id: str, llm, sandbox, task: Task, style: str, acceptance: list[str]) -> None:
    job = _runs[job_id]
    try:
        agent = HierarchicalAgent(llm, PRESETS["hmas-v0"], sandbox=sandbox)
        run = Run(task=task, dataset=style, cfg=agent.cfg)
        job["run"] = run
        job["status"], job["started"] = "running", time.time()
        # same call as HierarchicalAgent.solve, but we keep `run` so the page can follow it live
        agent.graph.invoke({"run": run, "cursor": 0, "repairs": 0}, config={"recursion_limit": 500})
        job["final_code"] = run.root.final.code if run.root and run.root.final else None
        if acceptance and job["final_code"]:
            rep = run_tests(sandbox, job["final_code"], acceptance, run.cfg.per_test_timeout)
            job["acceptance_results"] = {
                "load_error": rep.load_error, "sandbox_error": rep.sandbox_error,
                "passed": rep.passed, "total": rep.total,
                "results": [{"test": x["test"], "ok": x["ok"], "error": x["error"]} for x in rep.results],
            }
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        job["finished"] = time.time()
        _save(job)


def _save(job: dict) -> None:
    try:
        HISTORY.mkdir(parents=True, exist_ok=True)
        (HISTORY / f"{job['id']}.json").write_text(json.dumps(_snapshot(job), ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------- API

@app.get("/api/status")
def status():
    sb = _docker()
    return {
        "docker": sb is not None,
        "local_runner": LocalSandbox.available(),
        "models": _model_status(),
        "framework": "hmas-v0",
        "config": PRESETS["hmas-v0"].as_dict(),
    }


@app.get("/api/examples")
def examples():
    return public_examples()


@app.post("/api/runs")
def create_run(req: RunRequest):
    task, style = _build_task(req)
    acceptance = parse_assert_block(req.acceptance, 50) if req.acceptance.strip() else []
    if req.acceptance.strip() and not acceptance:
        raise HTTPException(400, "验收测试要写成 assert 语句，每行一条")
    if req.mode == "offline":
        if not req.example_id:
            raise HTTPException(400, "离线演示只能运行内置示例；自定义题目请切换到“真实模型”模式")
        llm, model_name = ScriptedLLM(BY_ID[req.example_id]), "离线脚本"
    else:
        if not req.confirm_paid:
            raise HTTPException(400, "真实模型会调用付费 API，请先勾选确认")
        if req.model not in LIVE_MODELS:
            raise HTTPException(400, f"未知模型 {req.model}")
        from hmas.llm import LLMClient
        cfg = load_model(req.model)
        try:
            cfg.api_key
        except RuntimeError as e:
            raise HTTPException(400, f"没有找到密钥：请在仓库根目录的 .env 里填写 {cfg.api_key_env}（{e}）")
        llm, model_name = LLMClient(cfg), req.model
    sandbox, sb_kind = _sandbox_for(req.mode)

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    title = BY_ID[req.example_id]["title"] if req.example_id else f"自定义题目：{task.entry_point}"
    job = {"id": job_id, "status": "queued", "mode": req.mode, "model": model_name, "title": title,
           "style": style, "entry_point": task.entry_point, "prompt": task.prompt, "created": time.time(),
           "started": None, "finished": None, "error": None, "sandbox": sb_kind, "run": None,
           "acceptance_results": None, "final_code": None}
    with _lock:
        _runs[job_id] = job
    _pool.submit(_execute, job_id, llm, sandbox, task, style, acceptance)
    return {"id": job_id}


@app.get("/api/runs")
def list_runs():
    items = {}
    if HISTORY.exists():
        for p in HISTORY.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items[d["id"]] = d
            except Exception:
                continue
    for jid, job in list(_runs.items()):
        items[jid] = _snapshot(job)
    keys = ("id", "status", "mode", "model", "title", "created", "elapsed")
    out = [{k: d.get(k) for k in keys} | {"passed": _root_passed(d)} for d in items.values()]
    return sorted(out, key=lambda d: d["created"] or 0, reverse=True)[:50]


def _root_passed(d: dict):
    roots = [n for n in d.get("nodes") or [] if n["root"]]
    return roots[0]["status"] == "passed" if roots and roots[0]["status"] in ("passed", "failed") else None


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    job = _runs.get(job_id)
    if job is not None:
        return _snapshot(job)
    p = HISTORY / f"{job_id}.json"
    if re.fullmatch(r"[0-9A-Za-z-]+", job_id) and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise HTTPException(404, "没有这次运行")


# ---------------------------------------------------------------- static page

if (STATIC / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description="HMAS demo web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    print(f"HMAS 演示系统: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
