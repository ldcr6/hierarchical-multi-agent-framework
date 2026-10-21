"""Run a candidate program against a list of assert tests inside the Docker sandbox.

The harness script below is executed by `python main.py` in a fresh container
(no network, read-only filesystem, resource limits; see hmas/sandbox.py). Each
test runs on its own with a per-test time limit, so one wrong or slow test does
not hide the others. For `assert a == b` tests the harness evaluates both sides
and reports the actual value, which becomes the repair feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..sandbox import DockerSandbox

HARNESS = r'''
import ast, json, math, signal, sys

def _timeout(signum, frame):
    raise TimeoutError("time limit exceeded")

signal.signal(signal.SIGALRM, _timeout)
PER_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and type(a) == type(b):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b

def _short(v, n=300):
    try:
        s = repr(v)
    except Exception as e:
        s = f"<unrepresentable {type(v).__name__}>"
    return s if len(s) <= n else s[:n] + "..."

out = {"load_error": None, "results": []}
src = open("candidate.py", encoding="utf-8").read()
tests = json.load(open("tests.json", encoding="utf-8"))
ns = {"__name__": "candidate"}
try:
    signal.alarm(PER_TEST * 2)
    exec(compile(src, "candidate.py", "exec"), ns)
    signal.alarm(0)
except BaseException as e:
    signal.alarm(0)
    out["load_error"] = f"{type(e).__name__}: {e}"

if out["load_error"] is None:
    base = dict(ns)
    base["math"] = math
    base["sys"] = sys
    for t in tests:
        r = {"test": t, "ok": False, "error": None}
        g = dict(base)
        try:
            node = ast.parse(t).body[0]
            signal.alarm(PER_TEST)
            if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                    and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
                left = eval(compile(ast.Expression(node.test.left), "<test>", "eval"), g)
                right = eval(compile(ast.Expression(node.test.comparators[0]), "<test>", "eval"), g)
                signal.alarm(0)
                r["ok"] = bool(_eq(left, right))
                if not r["ok"]:
                    r["error"] = f"expected {_short(right)}, got {_short(left)}"
            else:
                exec(compile(t, "<test>", "exec"), g)
                signal.alarm(0)
                r["ok"] = True
        except BaseException as e:
            signal.alarm(0)
            r["error"] = f"{type(e).__name__}: {_short(e)}" if not isinstance(e, AssertionError) else "AssertionError"
        out["results"].append(r)

print("@@HMAS@@" + json.dumps(out))
'''


@dataclass
class TestReport:
    load_error: str | None = None
    results: list[dict] = field(default_factory=list)
    sandbox_error: str | None = None   # container failed / timed out as a whole

    @property
    def passed(self) -> int:
        return sum(r["ok"] for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> bool:
        return self.load_error is None and self.sandbox_error is None and self.passed == self.total

    def failures(self) -> list[dict]:
        return [r for r in self.results if not r["ok"]]


def run_tests(sandbox: DockerSandbox, program: str, tests: list[str], per_test: int = 3) -> TestReport:
    """Execute `program`, then each test, in one sandbox container."""
    files = {"candidate.py": program, "tests.json": json.dumps(tests)}
    timeout = 10 + per_test * (len(tests) + 2)
    r = sandbox.run_python(HARNESS, stdin_files=files, args=[str(per_test)], timeout=timeout)
    marker = "@@HMAS@@"
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith(marker)), None)
    if line is None:
        why = "timed out" if r.timed_out else f"exit {r.returncode}"
        tail = (r.stderr or r.stdout)[-500:]
        return TestReport(sandbox_error=f"sandbox run failed ({why}): {tail}")
    data = json.loads(line[len(marker):])
    return TestReport(load_error=data["load_error"], results=data["results"])
