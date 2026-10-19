"""End-to-end plumbing test: run.py -> real OpenAI client -> a local mock server.

Checks that requests, usage accounting and output files work without an API key.
The mock answers with the reference solution; this is NOT a model score.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hmas.bench import run as run_mod
from hmas.bench.datasets import dataset_path, load_tasks

pytestmark = pytest.mark.skipif(not dataset_path("humaneval").exists(), reason="dataset not downloaded")


def test_run_against_mock_server(tmp_path, monkeypatch):
    tasks = {t.prompt.strip(): t for t in load_tasks("humaneval")[:3]}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msg = body["messages"][-1]["content"]
            t = next(t for p, t in tasks.items() if p in msg)
            content = f"```python\n{t.prompt}{t.canonical_solution}```"
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": content}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = tmp_path / "models.yaml"
    cfg.write_text(f"models:\n  mock:\n    base_url: http://127.0.0.1:{srv.server_port}/v1\n"
                   f"    model: mock\n    api_key_env: MOCK_KEY\n")
    monkeypatch.setenv("MOCK_KEY", "x")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(run_mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_mod, "load_model", lambda name: __import__("hmas.config", fromlist=["x"]).load_model(name, cfg))
    import hmas.llm as llm
    monkeypatch.setattr(llm, "DEFAULT_CACHE", tmp_path / "cache.sqlite")
    monkeypatch.setattr(llm.LLMClient.__init__, "__defaults__", (tmp_path / "cache.sqlite", 5))

    assert run_mod.main(["--dataset", "humaneval", "--model", "mock", "--limit", "3", "--run-name", "t"]) == 0
    out = tmp_path / "results" / "t"
    recs = [json.loads(l) for l in (out / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 3 and all(r["usage"]["calls"] == 1 and r["usage"]["prompt_tokens"] == 100 for r in recs)
    assert len((out / "samples.jsonl").read_text().splitlines()) == 3
    srv.shutdown()
