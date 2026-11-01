"""Demo web backend: offline mode end to end, and live mode through the real OpenAI client
against a local mock server (no API key, no network). Programs come from webapp/examples.py."""
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import webapp.server as server  # noqa: E402
from webapp.examples import BY_ID, ScriptedLLM  # noqa: E402
from webapp.local_sandbox import LocalSandbox  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="local test runner needs SIGALRM")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "HISTORY", tmp_path / "runs")
    monkeypatch.setattr(server, "_docker", lambda: None)
    return TestClient(server.app)


def wait(client, rid, limit=60):
    t0 = time.time()
    while time.time() - t0 < limit:
        d = client.get(f"/api/runs/{rid}").json()
        if d["status"] in ("done", "error"):
            return d
        time.sleep(0.2)
    raise AssertionError("run did not finish")


@pytest.fixture(autouse=True)
def fast_script(monkeypatch):
    orig = ScriptedLLM.__init__
    monkeypatch.setattr(ScriptedLLM, "__init__", lambda self, ex, delay=0.0: orig(self, ex, 0.0))


@pytest.mark.parametrize("eid", list(BY_ID))
def test_offline_examples(client, eid):
    ex = BY_ID[eid]
    rid = client.post("/api/runs", json={"mode": "offline", "example_id": eid,
                                         "acceptance": ex["acceptance"]}).json()["id"]
    d = wait(client, rid)
    assert d["status"] == "done", d["error"]
    assert all(n["status"] == "passed" for n in d["nodes"])
    acc = d["acceptance_results"]
    assert acc["passed"] == acc["total"] > 0
    assert any(r["id"] == rid for r in client.get("/api/runs").json())


def test_prime_fib_shows_a_repair(client):
    rid = client.post("/api/runs", json={"mode": "offline", "example_id": "prime_fib"}).json()["id"]
    d = wait(client, rid)
    node = next(n for n in d["nodes"] if n["name"] == "is_prime")
    assert [c["passed"] for c in node["candidates"]] == [False, True]
    assert node["final_index"] == 1


def test_input_checks(client):
    assert client.post("/api/runs", json={"mode": "offline", "style": "humaneval",
                                          "prompt": "def f(x):\n    '''x'''\n"}).status_code == 400
    assert client.post("/api/runs", json={"mode": "live", "example_id": "prime_fib"}).status_code == 400
    # live mode never runs model code outside Docker
    r = client.post("/api/runs", json={"mode": "live", "example_id": "prime_fib", "confirm_paid": True})
    assert r.status_code == 400


def test_live_mode_through_openai_client(client, monkeypatch, tmp_path):
    script = ScriptedLLM(BY_ID["top_k_words"], delay=0.0)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = script._reply(body["messages"][0]["content"], body["messages"][-1]["content"])
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": text}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    real_load = server.load_model

    def fake_load(name):
        cfg = real_load(name)
        cfg.base_url = f"http://127.0.0.1:{srv.server_port}"
        return cfg

    monkeypatch.setattr(server, "load_model", fake_load)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(server, "_docker", lambda: LocalSandbox())   # stands in for Docker in this test
    ex = BY_ID["top_k_words"]
    rid = client.post("/api/runs", json={"mode": "live", "style": "mbpp", "prompt": ex["prompt"],
                                         "entry_point": "top_k_words", "confirm_paid": True,
                                         "acceptance": ex["acceptance"]}).json()["id"]
    d = wait(client, rid)
    srv.shutdown()
    assert d["status"] == "done", d["error"]
    assert d["usage"]["total"]["calls"] == 9
    assert d["acceptance_results"]["passed"] == 2
