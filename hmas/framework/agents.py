"""The three agents: Planner (P), Coder (C) and Verifier (V).

Each agent is a thin role around the shared LLMClient; usage is accounted per role
so the paper can report where the tokens go (planning vs. coding vs. verification).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..bench.datasets import Task
from ..codeutil import extract_code
from ..llm import LLMClient, Usage
from ..sandbox import DockerSandbox
from . import prompts
from .config import FrameworkConfig
from .harness import run_tests
from .program import (assemble, merge_programs, parse_assert_block, parse_confidence, parse_ok, prompt_imports,
                      split_code, visible_tests)
from .tree import Candidate, Node, parse_plan

ROLES = ("planner", "coder", "verifier")


@dataclass
class Run:
    """Everything one problem's solve produces; shared by the agents and the graph nodes."""
    task: Task
    dataset: str
    cfg: FrameworkConfig
    usage: dict[str, Usage] = field(default_factory=lambda: {r: Usage() for r in ROLES})
    trace: list[dict] = field(default_factory=list)
    helpers: list[Node] = field(default_factory=list)
    root: Node | None = None
    composition: str = ""
    plan_problems: list[str] = field(default_factory=list)
    base_imports: list[str] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        return self.helpers + ([self.root] if self.root else [])

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.usage.values())

    def over_budget(self) -> bool:
        return self.calls >= self.cfg.max_calls

    def total_usage(self) -> Usage:
        t = Usage()
        for u in self.usage.values():
            t.calls += u.calls
            t.prompt_tokens += u.prompt_tokens
            t.completion_tokens += u.completion_tokens
            t.cached_calls += u.cached_calls
            t.latency += u.latency
        return t

    def done_helpers_code(self, upto: Node | None = None) -> tuple[list[str], list[str]]:
        """(imports, code parts) of helpers finished before `upto`."""
        imports, parts = list(self.base_imports), []
        for h in self.helpers:
            if h is upto:
                break
            # a helper whose best version still does not parse is left out (Mbpp/245: the
            # first draft was cut off at max_tokens and both repairs had syntax errors)
            if h.final is None or not parse_ok(h.final.code)[0]:
                continue
            imp, body = split_code(h.final.code)
            imports += imp
            parts += body
        return imports, parts


class _Role:
    role = ""
    system = ""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def ask(self, run: Run, step: str, user: str, **trace_extra) -> str:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        r = self.llm.chat(messages, usage=run.usage[self.role])
        run.trace.append({"step": step, "role": self.role, "prompt_tokens": r.prompt_tokens,
                          "completion_tokens": r.completion_tokens, "response": r.content, **trace_extra})
        return r.content


class Planner(_Role):
    role, system = "planner", prompts.PLANNER_SYSTEM

    def decompose(self, run: Run) -> None:
        t, cfg = run.task, run.cfg
        text = self.ask(run, "plan", prompts.planner_decompose(t, run.dataset, cfg.max_subtasks))
        helpers, root_k, comp, problems = parse_plan(text, t.entry_point, cfg.max_subtasks)
        run.helpers, run.composition, run.plan_problems = helpers, comp, problems
        run.root = Node(name=t.entry_point, signature=t.entry_point, description="", is_root=True,
                        complexity=root_k, depends_on=[h.name for h in helpers])
        for h in helpers:
            h.level = cfg.leaf_level
        run.root.level = cfg.root_level
        run.root.visible = visible_tests(t, run.dataset)
        run.base_imports = prompt_imports(t, run.dataset)
        if problems:
            run.trace.append({"step": "plan_check", "problems": problems})


class Coder(_Role):
    role, system = "coder", prompts.CODER_SYSTEM

    def _clean_leaf(self, run: Run, node: Node, text: str) -> str:
        code = extract_code(text)
        ok, _ = parse_ok(code)
        if not ok:
            return code  # the static check reports the syntax error
        other = {h.name for h in run.helpers if h is not node} | {run.task.entry_point}
        imp, body = split_code(code, drop_defs=other)
        return assemble(body, imp)

    def _clean_program(self, run: Run, text: str, drop_helpers: bool) -> str:
        """Root code -> full program: prompt imports + finished helpers + this code."""
        code = extract_code(text)
        ok, _ = parse_ok(code)
        if not ok:
            return code
        imports, parts = run.done_helpers_code() if drop_helpers else (list(run.base_imports), [])
        drop = ({h.name for h in run.helpers if h.final is not None and parse_ok(h.final.code)[0]}
                if drop_helpers else set())
        imp, body = split_code(code, drop_defs=drop)
        return assemble(parts + body, imports + imp)

    def implement(self, run: Run, node: Node) -> Candidate:
        t, ds = run.task, run.dataset
        if not node.is_root:
            _, parts = run.done_helpers_code(upto=node)
            text = self.ask(run, "implement", prompts.coder_implement_leaf(t, ds, node, "\n\n".join(parts)),
                            node=node.name)
            node.confidence = parse_confidence(text)
            return Candidate(self._clean_leaf(run, node, text), "implement")
        done = [h for h in run.helpers if h.final is not None]
        if done:
            _, parts = run.done_helpers_code()
            text = self.ask(run, "compose", prompts.coder_compose_root(t, ds, "\n\n".join(parts), run.composition),
                            node=node.name)
            origin, drop = "compose", True
        else:
            text = self.ask(run, "implement", prompts.coder_implement_root(t, ds), node=node.name)
            origin, drop = "implement", False
        node.confidence = parse_confidence(text)
        return Candidate(self._clean_program(run, text, drop_helpers=drop), origin)

    def repair(self, run: Run, node: Node, cand: Candidate) -> Candidate:
        t, ds = run.task, run.dataset
        origin = f"repair-{len(node.candidates)}"
        if not node.is_root:
            _, parts = run.done_helpers_code(upto=node)
            text = self.ask(run, "repair", prompts.coder_repair_leaf(t, ds, node, "\n\n".join(parts), cand),
                            node=node.name)
            return Candidate(self._clean_leaf(run, node, text), origin)
        text = self.ask(run, "repair", prompts.coder_repair_root(t, ds, cand), node=node.name)
        # a root repair may fix any function; functions it leaves out are kept from `cand`
        new = extract_code(text)
        if not parse_ok(new)[0]:
            return Candidate(new, origin)
        return Candidate(merge_programs(cand.code, new, run.base_imports), origin)


class Verifier(_Role):
    role, system = "verifier", prompts.VERIFIER_SYSTEM

    def __init__(self, llm: LLMClient, sandbox: DockerSandbox):
        super().__init__(llm)
        self.sandbox = sandbox

    # L1 ------------------------------------------------------------
    @staticmethod
    def static_check(node: Node, code: str) -> str | None:
        ok, err = parse_ok(code)
        if not ok:
            return err
        tree = ast.parse(code)
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if node.name not in funcs:
            return f"function `{node.name}` is not defined"
        for f in funcs.values():
            if not any(isinstance(x, ast.Return) for x in ast.walk(f)):
                return f"function `{f.name}` has no return statement"
        if not node.is_root:
            want = ast.parse(node.signature + ":\n    pass\n").body[0].args
            got = funcs[node.name].args
            if len(want.args) != len(got.args):
                return (f"`{node.name}` takes {len(got.args)} positional parameters, "
                        f"the plan says {len(want.args)} (`{node.signature}`)")
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as e:
            return f"SyntaxError: {e.msg}"
        return None

    # test generation ------------------------------------------------
    def write_tests(self, run: Run, node: Node) -> list[str]:
        if node.is_root:
            spec = prompts.problem_text(run.task, run.dataset)
            sig = run.task.entry_point
        else:
            spec, sig = node.description, node.signature
        text = self.ask(run, "write_tests", prompts.verifier_tests(spec, sig, run.cfg.n_generated_tests),
                        node=node.name)
        tests = parse_assert_block(extract_code(text), run.cfg.n_generated_tests)
        # keep only tests that call this node's function and are not copies of visible tests
        return [x for x in tests if node.name in x and x not in node.visible]

    # verify at a fixed level ----------------------------------------
    def verify(self, run: Run, node: Node, cand: Candidate) -> Candidate:
        level = node.level
        if level <= 0:
            return cand
        code = cand.code if node.is_root else self._leaf_program(run, node, cand)
        cand.static_error = self.static_check(node, cand.code if not node.is_root else code)
        if cand.static_error or level == 1:
            run.trace.append({"step": "verify", "node": node.name, "level": level, "origin": cand.origin,
                              "static_error": cand.static_error, "passed": cand.passed})
            return cand
        if node.tests is None:
            node.tests = [] if run.over_budget() else self.write_tests(run, node)
        tests = node.visible + node.tests
        cand.n_visible = len(node.visible)
        cand.report = run_tests(self.sandbox, code, tests, run.cfg.per_test_timeout)
        r = cand.report
        run.trace.append({"step": "verify", "node": node.name, "level": level, "origin": cand.origin,
                          "passed": cand.passed, "tests_passed": r.passed, "tests_total": r.total,
                          "load_error": r.load_error, "sandbox_error": r.sandbox_error,
                          "failures": [f"{x['test']} -> {x['error']}" for x in r.failures()][:8]})
        return cand

    @staticmethod
    def _leaf_program(run: Run, node: Node, cand: Candidate) -> str:
        imports, parts = run.done_helpers_code(upto=node)
        ok, _ = parse_ok(cand.code)
        if not ok:
            return cand.code
        imp, body = split_code(cand.code)
        return assemble(parts + body, imports + imp)
