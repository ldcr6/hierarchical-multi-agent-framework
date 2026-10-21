"""Program-text utilities: visible-test extraction, splitting code into parts, assembling programs.

Everything here is static (ast only); nothing executes model code on the host.
"""
from __future__ import annotations

import ast
import doctest
import re

from ..bench.datasets import Task


# ---------------------------------------------------------------- visible tests

def _is_expr(src: str) -> bool:
    try:
        ast.parse(src.strip(), mode="eval")
        return True
    except SyntaxError:
        return False


def visible_tests(task: Task, dataset: str) -> list[str]:
    """Tests that are part of the problem statement (T^vis), as assert statements.

    HumanEval+: doctest examples in the docstring (`>>> f(x)` followed by the
    expected value) and informal examples such as `f(x) => y` or `f(x) ➞ y`.
    MBPP+: the `assert` line(s) in the prompt. Hidden EvalPlus tests are never used.
    """
    tests: list[str] = []

    def add(t: str) -> None:
        # e.g. `is_happy(a) => False` in HumanEval/80 uses an unquoted string: not a test
        if _parses(t) and t not in tests and _free_names(t) <= {task.entry_point} | _BUILTINS:
            tests.append(t)

    if dataset == "mbpp":
        for line in task.prompt.splitlines():
            s = line.strip()
            if s.startswith("assert "):
                add(s)
        return tests

    try:
        examples = doctest.DocTestParser().get_examples(task.prompt)
    except ValueError:  # malformed doctest block
        examples = []
    for ex in examples:
        src = ex.source.strip()
        # the docstring's closing quotes end up in the last example's expected output
        lines = [ln for ln in ex.want.strip().splitlines() if ln.strip() not in ('"""', "'''")]
        want = " ".join(" ".join(lines).split())
        if want and _is_expr(src) and _is_expr(want):  # skip printed text that is not a value
            add(f"assert ({src}) == ({want})")

    name = re.escape(task.entry_point)
    for line in task.prompt.splitlines():
        s = line.strip().rstrip(".,;")
        if not re.match(rf"{name}\s*\(", s):
            continue
        for m in _ARROW.finditer(s):
            lhs, rhs = s[:m.start()].strip(), s[m.end():].strip()
            if lhs.endswith(")") and _is_expr(lhs) and _is_expr(rhs):
                add(f"assert ({lhs}) == ({rhs})")
                break
    return tests


# the test harness pre-imports math and sys (MBPP+ asserts use math.isclose, sys.getsizeof)
_BUILTINS = set(dir(__import__("builtins"))) | {"math", "sys"}


def _free_names(src: str) -> set[str]:
    tree = ast.parse(src)
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.Lambda) for a in n.args.args}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - bound


# plain "=" is not an arrow: HumanEval/130 writes the recurrence as `tri(4) = 3`
_ARROW = re.compile(r"\s*(?:=>|➞|==|->|→|should return|returns|return)\s*")


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------- code parts

def parse_ok(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def defined_functions(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def split_code(code: str, drop_defs: set[str] = frozenset()) -> tuple[list[str], list[str]]:
    """Split a code block into (import lines, other top-level definitions).

    Keeps functions, classes and module-level assignments; drops top-level calls,
    prints and `if __name__ == "__main__"` blocks, and any function named in
    `drop_defs` (helpers that were already implemented and verified elsewhere).
    """
    tree = ast.parse(code)
    imports, body = [], []
    for node in tree.body:
        seg = ast.get_source_segment(code, node)
        if seg is None:
            seg = ast.unparse(node)
        # keep decorators with their function
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
            start = node.decorator_list[0].lineno
            seg = "\n".join(code.splitlines()[start - 1: node.end_lineno])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in drop_defs:
                body.append(seg)
        elif isinstance(node, (ast.ClassDef, ast.Assign, ast.AnnAssign)):
            body.append(seg)
    return imports, body


def merge_programs(old: str, new: str, imports: list[str] = ()) -> str:
    """Replace `old`'s top-level definitions by same-named ones from `new`; keep the rest, add new ones."""
    def named_parts(code: str) -> tuple[list[str], list[tuple[str | None, str]]]:
        imp, body = split_code(code)
        out = []
        for seg in body:
            node = ast.parse(seg).body[-1] if seg.strip() else None
            name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
            out.append((name, seg))
        return imp, out

    old_imp, old_parts = named_parts(old)
    new_imp, new_parts = named_parts(new)
    replaced = {n: s for n, s in new_parts if n}
    merged = [replaced.pop(n, s) if n else s for n, s in old_parts]
    merged += [s for n, s in new_parts if n is None or n in replaced]
    return assemble(merged, list(imports) + old_imp + new_imp)


def prompt_imports(task: Task, dataset: str) -> list[str]:
    """Import lines that the problem statement itself provides (HumanEval: `from typing import List`)."""
    if dataset != "humaneval":
        return []
    try:
        tree = ast.parse(task.prompt + "    pass\n")
    except SyntaxError:
        return []
    return [ast.get_source_segment(task.prompt + "    pass\n", n) for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))]


def assemble(parts: list[str], imports: list[str] = ()) -> str:
    """Join imports (deduplicated, in order) and code parts into one program."""
    seen, imp = set(), []
    for line in imports:
        if line not in seen:
            seen.add(line)
            imp.append(line)
    chunks = (["\n".join(imp)] if imp else []) + [p.strip("\n") for p in parts if p.strip()]
    return "\n\n\n".join(chunks) + "\n"


_CONF = re.compile(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", re.I)


def parse_confidence(text: str) -> float | None:
    m = _CONF.search(text)
    if not m:
        return None
    v = float(m.group(1))
    return v if 0.0 <= v <= 1.0 else None


def parse_assert_block(code: str, limit: int) -> list[str]:
    """Pull individual assert statements out of a model-written test block."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # fall back to line-by-line
        out = []
        for line in code.splitlines():
            s = line.strip()
            if s.startswith("assert ") and _parses(s):
                out.append(s)
        return out[:limit]
    out = [ast.unparse(n) for n in tree.body if isinstance(n, ast.Assert)]
    return out[:limit]
