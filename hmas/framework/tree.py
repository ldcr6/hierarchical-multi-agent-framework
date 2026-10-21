"""Task decomposition tree: nodes, plan parsing, topological order."""
from __future__ import annotations

import ast
import json
import keyword
import re
from dataclasses import dataclass, field

from .harness import TestReport


@dataclass
class Candidate:
    code: str                        # leaf: the helper's code block; root: the full program
    origin: str                      # "implement" | "compose" | "repair-1" | ...
    static_error: str | None = None
    report: TestReport | None = None  # None when the level did not run tests
    n_visible: int = 0               # how many of the report's tests are visible (problem) tests

    def score(self) -> tuple:
        """Higher is better: static OK, loads, visible tests passed, generated tests passed."""
        if self.static_error:
            return (0, 0, 0, 0)
        if self.report is None:
            return (1, 1, 0, 0)
        r = self.report
        if r.sandbox_error or r.load_error:
            return (1, 0, 0, 0)
        vis = sum(x["ok"] for x in r.results[: self.n_visible])
        gen = sum(x["ok"] for x in r.results[self.n_visible:])
        return (1, 1, vis, gen)

    @property
    def passed(self) -> bool:
        if self.static_error:
            return False
        return self.report is None or self.report.ok


@dataclass
class Node:
    name: str                         # function name
    signature: str                    # "def f(x: int) -> int"
    description: str
    depends_on: list[str] = field(default_factory=list)
    is_root: bool = False
    complexity: float | None = None   # κ(n): Planner's 1–5 rating mapped to [0, 1]
    confidence: float | None = None   # ĉ(n): Coder's self-reported confidence of the first draft
    level: int = 2                    # verification level used for this node
    tests: list[str] | None = None    # generated tests (written once, reused across repairs)
    visible: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    final: Candidate | None = None

    def best(self) -> Candidate:
        # max() keeps the first of equal scores, so the earliest version wins ties
        return max(self.candidates, key=lambda c: c.score())

    def summary(self) -> dict:
        first = self.candidates[0] if self.candidates else None
        return {
            "name": self.name, "root": self.is_root, "signature": self.signature,
            "depends_on": self.depends_on, "complexity": self.complexity, "confidence": self.confidence,
            "level": self.level, "n_visible": len(self.visible), "n_generated": len(self.tests or []),
            "attempts": len(self.candidates),
            "first_passed": None if first is None else first.passed,
            "final_passed": None if self.final is None else self.final.passed,
            "final_origin": None if self.final is None else self.final.origin,
            "final_score": None if self.final is None else list(self.final.score()),
        }


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _norm_complexity(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return min(max((x - 1) / 4, 0.0), 1.0)


def extract_json(text: str) -> dict | None:
    """First JSON object in a model response (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


def _signature_ok(sig: str, name: str) -> bool:
    src = sig.strip().rstrip(":") + ":\n    pass\n"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == name)


def parse_plan(text: str, entry_point: str, max_subtasks: int) -> tuple[list[Node], float | None, str, list[str]]:
    """Validate the Planner's JSON. Returns (subtasks in topological order, κ(root), composition note, problems)."""
    problems: list[str] = []
    data = extract_json(text)
    if data is None:
        return [], None, "", ["plan is not valid JSON"]
    root_k = _norm_complexity(data.get("complexity"))
    comp = str(data.get("composition", "") or "")
    raw = data.get("subtasks") or []
    if not isinstance(raw, list):
        return [], root_k, comp, ["subtasks is not a list"]

    nodes: list[Node] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        sig = str(item.get("signature", "")).strip()
        if not sig.startswith("def "):
            sig = f"def {sig}" if sig.startswith(name + "(") else sig
        if (not _IDENT.match(name) or keyword.iskeyword(name) or name == entry_point
                or name in seen or not _signature_ok(sig, name)):
            problems.append(f"dropped subtask {name!r}")
            continue
        deps = item.get("depends_on") or []
        nodes.append(Node(name=name, signature=sig.rstrip(":"),
                          description=str(item.get("description", "")).strip(),
                          depends_on=[str(d) for d in deps if isinstance(d, str)] if isinstance(deps, list) else [],
                          complexity=_norm_complexity(item.get("complexity"))))
        seen.add(name)
        if len(nodes) == max_subtasks:
            break
    for n in nodes:
        n.depends_on = [d for d in n.depends_on if d in seen and d != n.name]
    return topo_order(nodes), root_k, comp, problems


def topo_order(nodes: list[Node]) -> list[Node]:
    """Dependencies first; keeps the Planner's order otherwise. Cycles are broken by dropping back edges."""
    by_name = {n.name: n for n in nodes}
    out, state = [], {}

    def visit(n: Node, stack: set[str]):
        if state.get(n.name) == "done":
            return
        stack.add(n.name)
        for d in list(n.depends_on):
            if d in stack:          # cycle: drop this edge
                n.depends_on.remove(d)
                continue
            visit(by_name[d], stack)
        stack.discard(n.name)
        state[n.name] = "done"
        out.append(n)

    for n in nodes:
        visit(n, set())
    return out
