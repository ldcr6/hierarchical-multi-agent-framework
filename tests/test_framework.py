"""Framework v0 (Planner + Coder + Verifier) with a scripted fake LLM.

The sandbox is replaced by a local subprocess runner so the control flow can be
tested without Docker; the test programs here are our own, never model output.
"""
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from hmas.bench.datasets import Task, dataset_path, load_tasks
from hmas.framework.agents import Verifier
from hmas.framework.config import FrameworkConfig
from hmas.framework.graph import HierarchicalAgent
from hmas.framework.harness import run_tests
from hmas.framework.program import merge_programs, visible_tests
from hmas.framework.tree import Node, parse_plan
from hmas.llm import LLMResponse
from hmas.sandbox import ExecResult


class LocalRunner:
    """Stand-in for DockerSandbox.run_python (tests only)."""

    def run_python(self, code, *, stdin_files=None, timeout=None, args=()):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").write_text(code, encoding="utf-8")
            for k, v in (stdin_files or {}).items():
                Path(d, k).write_text(v, encoding="utf-8")
            p = subprocess.run([sys.executable, "main.py", *args], cwd=d, capture_output=True, text=True,
                               timeout=timeout)
            return ExecResult(p.returncode, p.stdout, p.stderr, False, 0.0)


# The harness limits each test with SIGALRM, which only exists on Unix. Windows hosts
# run these tests through the real Docker sandbox instead (the harness always runs
# inside a Linux container in real runs).
HAS_ALARM = hasattr(signal, "SIGALRM")


def runner():
    if HAS_ALARM:
        return LocalRunner()
    from hmas.sandbox import DockerSandbox
    sb = DockerSandbox(timeout=60)
    if not (sb.available() and sb.image_exists()):
        pytest.skip("no SIGALRM on this host and Docker sandbox not available")
    return sb


TASK = Task("HumanEval/x", 'def add_then_double(a, b):\n    """Return 2 * (a + b).\n'
            '    >>> add_then_double(1, 2)\n    6\n    """\n', "add_then_double", "")

PLAN = {"complexity": 2, "composition": "double the sum",
        "subtasks": [{"name": "add", "signature": "def add(a, b)", "description": "a + b",
                      "depends_on": [], "complexity": 1},
                     {"name": "double", "signature": "def double(x)", "description": "2 * x",
                      "depends_on": [], "complexity": 1}]}


class ScriptedLLM:
    """Answers by role (system prompt) and step; the first `add` is buggy."""

    def __init__(self):
        self.calls = []
        self.add_attempts = 0

    def chat(self, messages, usage=None, **kw):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "Planner" in system:
            out = "```json\n" + json.dumps(PLAN) + "\n```"
        elif "Tester" in system:
            if "`def add(a, b)`" in user:
                out = "```python\nassert add(1, 2) == 3\nassert add(0, 0) == 0\n```"
            elif "`def double(x)`" in user:
                out = "```python\nassert double(4) == 8\n```"
            else:
                out = "```python\nassert add_then_double(2, 3) == 10\n```"
        elif "Implement this helper" in user or "Helper function:" in user:
            if "`def add(a, b)`" in user:
                self.add_attempts += 1
                body = "a - b" if self.add_attempts == 1 else "a + b"
                out = f"```python\ndef add(a, b):\n    return {body}\n```\nCONFIDENCE: 0.9"
            else:
                out = "```python\ndef double(x):\n    return 2 * x\n\nprint(double(2))\n```\nCONFIDENCE: 0.8"
        else:  # compose the entry function
            out = ("```python\ndef add(a, b):\n    return 0\n\n"
                   "def add_then_double(a, b):\n    return double(add(a, b))\n```\nCONFIDENCE: 0.7")
        self.calls.append(out)
        r = LLMResponse(out, 10, 5, 0.0)
        if usage is not None:
            usage.add(r)
        return r


def test_full_loop_repairs_buggy_helper():
    llm = ScriptedLLM()
    agent = HierarchicalAgent(llm, FrameworkConfig(), sandbox=runner())
    res = agent.solve(TASK, "humaneval")

    ns = {}
    exec(res.solution, ns)
    assert ns["add_then_double"](2, 3) == 10
    # the composer's redefinition of the verified `add` was dropped; stray print removed
    assert res.solution.count("def add(") == 1 and "print(" not in res.solution

    nodes = {n["name"]: n for n in res.extra["nodes"]}
    assert list(nodes) == ["add", "double", "add_then_double"]
    assert nodes["add"]["first_passed"] is False and nodes["add"]["final_passed"] is True
    assert nodes["add"]["final_origin"] == "repair-1"
    assert nodes["double"]["attempts"] == 1 and nodes["add_then_double"]["final_passed"]
    assert nodes["add"]["confidence"] == 0.9 and nodes["add"]["complexity"] == 0.0
    assert nodes["add_then_double"]["n_visible"] == 1

    roles = res.extra["usage_by_role"]
    # planner 1; coder: add, add-repair, double, compose = 4; verifier: 3 test-writing calls
    assert (roles["planner"]["calls"], roles["coder"]["calls"], roles["verifier"]["calls"]) == (1, 4, 3)
    assert res.usage.calls == 8


def test_level_zero_skips_verification():
    llm = ScriptedLLM()
    cfg = FrameworkConfig(leaf_level=0, root_level=0)
    res = HierarchicalAgent(llm, cfg, sandbox=runner()).solve(TASK, "humaneval")
    assert res.extra["usage_by_role"]["verifier"]["calls"] == 0
    assert res.usage.calls == 4  # plan + 2 helpers + compose, no repair (buggy add kept)


def test_bad_plan_falls_back_to_single_node():
    class NoPlan(ScriptedLLM):
        def chat(self, messages, usage=None, **kw):
            if "Planner" in messages[0]["content"]:
                r = LLMResponse("I cannot do that.", 1, 1, 0.0)
                usage.add(r)
                return r
            if "Write a complete solution" in messages[-1]["content"]:
                r = LLMResponse("```python\ndef add_then_double(a, b):\n    return 2 * (a + b)\n```", 1, 1, 0.0)
                usage.add(r)
                return r
            return super().chat(messages, usage, **kw)

    res = HierarchicalAgent(NoPlan(), FrameworkConfig(), sandbox=runner()).solve(TASK, "humaneval")
    assert res.extra["n_helpers"] == 0 and res.extra["plan_problems"]
    assert "return 2 * (a + b)" in res.solution


def test_parse_plan_validates_and_orders():
    text = json.dumps({"complexity": 5, "subtasks": [
        {"name": "b", "signature": "def b(x)", "depends_on": ["a"]},
        {"name": "a", "signature": "a(x)", "depends_on": ["b"]},       # cycle, missing "def"
        {"name": "f", "signature": "def f(x)"},                        # same name as entry point
        {"name": "bad name", "signature": "def bad name(x)"},
        {"name": "c", "signature": "def c(", "depends_on": []},        # unparsable signature
    ]})
    nodes, k, _, problems = parse_plan(text, "f", 5)
    assert [n.name for n in nodes] == ["a", "b"] and k == 1.0
    assert nodes[0].signature == "def a(x)" and len(problems) == 3
    assert parse_plan("no json here", "f", 5)[0] == []


def test_merge_programs_keeps_unrepaired_helpers():
    old = "import math\n\ndef h(x):\n    return x\n\ndef f(x):\n    return h(x) + 1\n"
    new = "def f(x):\n    return h(x) + 2\n\ndef g(x):\n    return x\n"
    out = merge_programs(old, new, ["from typing import List"])
    ns = {}
    exec(out, ns)
    assert ns["f"](1) == 3 and "g" in ns and out.startswith("from typing import List\nimport math")


def test_static_check():
    leaf = Node("g", "def g(x, y)", "")
    assert Verifier.static_check(leaf, "def g(x, y):\n    return x\n") is None
    assert "not defined" in Verifier.static_check(leaf, "def h(x, y):\n    return x\n")
    assert "no return" in Verifier.static_check(leaf, "def g(x, y):\n    yield x\n")
    assert "positional" in Verifier.static_check(leaf, "def g(x):\n    return x\n")
    assert "SyntaxError" in Verifier.static_check(leaf, "def g(x, y)\n    return x\n")


def test_harness_reports_values_and_errors():
    prog = "def f(x):\n    if x == 3:\n        raise ValueError('three')\n    return x * 2\n"
    tests = ["assert f(1) == 2", "assert f(2) == 5", "assert f(3) == 6", "assert f(0.1) == 0.2",
             "assert isinstance(f(1), int)"]
    rep = run_tests(runner(), prog, tests)
    assert [r["ok"] for r in rep.results] == [True, False, False, True, True]
    assert rep.results[1]["error"] == "expected 5, got 4"
    assert "ValueError" in rep.results[2]["error"]
    bad = run_tests(runner(), "def f(:\n", tests)
    assert bad.load_error and not bad.ok


def test_visible_tests_examples():
    t = Task("HumanEval/80", 'def is_happy(s):\n    """\n    is_happy(a) => False\n    is_happy("abc") => True\n'
             '    >>> is_happy("aa")\n    False\n    """\n', "is_happy", "")
    # unquoted `a` is not a valid test; doctest + arrow examples are kept
    assert visible_tests(t, "humaneval") == ['assert (is_happy("aa")) == (False)',
                                             'assert (is_happy("abc")) == (True)']
    m = Task("Mbpp/2", '"""\nWrite a function.\nassert similar(1) == 2\n"""\n', "similar", "")
    assert visible_tests(m, "mbpp") == ["assert similar(1) == 2"]


# The problem statement itself is wrong here: the docstring says median([-10, 4, 6, 1000, 10, 20])
# is 15.0, the reference solution (and the hidden tests) give 8.0. Every method sees the same text.
KNOWN_WRONG_EXAMPLES = {"HumanEval/47"}


@pytest.mark.skipif(not dataset_path("humaneval").exists(), reason="dataset not downloaded")
@pytest.mark.skipif(not HAS_ALARM, reason="needs SIGALRM (one sandbox container per task would be too slow)")
@pytest.mark.parametrize("ds", ["humaneval", "mbpp"])
def test_visible_tests_pass_reference_solutions(ds):
    """Every extracted visible test must pass on the reference solution (reference code is trusted)."""
    bad = []
    for t in load_tasks(ds):
        tests = visible_tests(t, ds)
        if not tests or t.task_id in KNOWN_WRONG_EXAMPLES:
            continue
        prog = t.prompt + t.canonical_solution if ds == "humaneval" else t.canonical_solution
        rep = run_tests(runner(), prog, tests, per_test=10)
        if not rep.ok:
            bad.append((t.task_id, rep.load_error, [r for r in rep.results if not r["ok"]][:2]))
    assert not bad, bad


def _docker_ready():
    from hmas.sandbox import DockerSandbox
    sb = DockerSandbox()
    return sb.available() and sb.image_exists()


@pytest.mark.skipif(not _docker_ready(), reason="Docker or sandbox image not available")
def test_full_loop_in_docker_sandbox():
    from hmas.sandbox import DockerSandbox
    res = HierarchicalAgent(ScriptedLLM(), FrameworkConfig(), sandbox=DockerSandbox(timeout=60)).solve(TASK, "humaneval")
    nodes = {n["name"]: n for n in res.extra["nodes"]}
    assert nodes["add"]["final_origin"] == "repair-1" and nodes["add_then_double"]["final_passed"]


def test_root_repair_returns_only_entry_function():
    class BuggyRoot(ScriptedLLM):
        def chat(self, messages, usage=None, **kw):
            user = messages[-1]["content"]
            if "Current program:" in user:  # root repair: answer with the entry function only
                out = "```python\ndef add_then_double(a, b):\n    return double(add(a, b))\n```"
            elif "Write the entry function" in user:
                out = "```python\ndef add_then_double(a, b):\n    return add(a, b)\n```"
            else:
                return super().chat(messages, usage, **kw)
            r = LLMResponse(out, 1, 1, 0.0)
            usage.add(r)
            return r

    res = HierarchicalAgent(BuggyRoot(), FrameworkConfig(), sandbox=runner()).solve(TASK, "humaneval")
    root = res.extra["nodes"][-1]
    assert root["first_passed"] is False and root["final_passed"] and root["final_origin"] == "repair-1"
    ns = {}
    exec(res.solution, ns)
    assert ns["add_then_double"](2, 3) == 10


def test_unparsable_helper_does_not_crash_the_root():
    """Mbpp/245: a helper whose every version has a syntax error must not crash composing the root."""
    class BrokenHelper(ScriptedLLM):
        def chat(self, messages, usage=None, **kw):
            user = messages[-1]["content"]
            if ("Implement this helper" in user or "Helper function:" in user) and "`def add(a, b)`" in user:
                out = "```python\ndef add(a, b:\n    return a +\n```"   # cut off / broken every time
            elif "Write the entry function" in user:
                out = ("```python\ndef add(a, b):\n    return a + b\n\n"
                       "def add_then_double(a, b):\n    return double(add(a, b))\n```")
            else:
                return super().chat(messages, usage, **kw)
            r = LLMResponse(out, 1, 1, 0.0)
            usage.add(r)
            return r

    res = HierarchicalAgent(BrokenHelper(), FrameworkConfig(), sandbox=runner()).solve(TASK, "humaneval")
    nodes = {n["name"]: n for n in res.extra["nodes"]}
    assert nodes["add"]["final_passed"] is False
    # identical broken repairs stop early: implement + 1 repair, not 2
    assert nodes["add"]["attempts"] == 2
    ns = {}
    exec(res.solution, ns)   # the root's own `add` is kept because the helper version was unusable
    assert ns["add_then_double"](2, 3) == 10
