"""Prompt templates for the Planner, Coder and Verifier.

All three agents use the same base model; only the role prompt and the inputs differ.
"""
from __future__ import annotations

from ..bench.datasets import Task


def problem_text(task: Task, dataset: str) -> str:
    if dataset == "humaneval":
        return (f"```python\n{task.prompt.strip()}\n```\n"
                f"The entry function is `{task.entry_point}`.")
    desc = task.prompt.strip().strip('"').strip()
    return (f"{desc}\n\nThe entry function must be named `{task.entry_point}`; "
            f"the example above shows how it is called.")


CODE_RULES = (
    "Rules: write plain Python 3 with only the standard library; no tests, example usage, "
    "print statements or `if __name__ == \"__main__\"` blocks; every function must end with "
    "an explicit `return` (no generator functions)."
)

CONFIDENCE_LINE = (
    "After the code block, write one line `CONFIDENCE: <number between 0 and 1>` giving the "
    "probability that your code is correct."
)

# ------------------------------------------------------------------ Planner

PLANNER_SYSTEM = (
    "You are the Planner in a multi-agent programming team. You break a programming problem "
    "into small helper functions that a Coder will implement one by one and that a Tester will "
    "test separately. You do not write the implementation."
)


def planner_decompose(task: Task, dataset: str, max_subtasks: int) -> str:
    return f"""Decompose the following problem into helper functions.

Problem:
{problem_text(task, dataset)}

Requirements:
- Give between 1 and {max_subtasks} helper functions. Each helper does one clear, testable job and is a pure function (inputs -> return value).
- Helper names must be valid Python identifiers and must differ from `{task.entry_point}`.
- `depends_on` lists other helpers (from this list) that the helper calls.
- The entry function `{task.entry_point}` itself is NOT a helper; describe in `composition` how it combines the helpers.
- Rate the difficulty of the whole problem and of each helper from 1 (trivial) to 5 (hard).

Answer with one JSON object in a ```json block and nothing else:
```json
{{
  "complexity": 2,
  "subtasks": [
    {{"name": "helper_name", "signature": "def helper_name(x: int) -> bool", "description": "what it returns, including edge cases", "depends_on": [], "complexity": 1}}
  ],
  "composition": "how {task.entry_point} uses the helpers"
}}
```"""


# ------------------------------------------------------------------ Coder

CODER_SYSTEM = (
    "You are the Coder in a multi-agent programming team. You implement exactly the function "
    "you are asked for, correctly and robustly, following the specification precisely."
)


def coder_implement_leaf(task: Task, dataset: str, node, deps_code: str) -> str:
    deps = (f"Helpers that are already implemented and tested (call them if useful, do not redefine them):\n"
            f"```python\n{deps_code.strip()}\n```\n\n") if deps_code.strip() else ""
    return f"""Overall problem (for context only):
{problem_text(task, dataset)}

Implement this helper function:
- Signature: `{node.signature}`
- Specification: {node.description}

{deps}Write the complete `{node.name}` function, plus any imports it needs, in one ```python block.
{CODE_RULES}
{CONFIDENCE_LINE}"""


def coder_compose_root(task: Task, dataset: str, helpers_code: str, composition: str) -> str:
    return f"""Problem:
{problem_text(task, dataset)}

These helper functions are already implemented and tested:
```python
{helpers_code.strip()}
```
Plan: {composition or "combine the helpers to solve the problem."}

Write the entry function `{task.entry_point}` that solves the problem by calling the helpers above
(do not redefine or copy the helpers), plus any imports it needs, in one ```python block.
{CODE_RULES}
{CONFIDENCE_LINE}"""


def coder_implement_root(task: Task, dataset: str) -> str:
    """Used only when the plan has no usable helpers."""
    return f"""Problem:
{problem_text(task, dataset)}

Write a complete solution (the function `{task.entry_point}` and anything it needs) in one ```python block.
{CODE_RULES}
{CONFIDENCE_LINE}"""


def _failures_text(cand) -> str:
    if cand.static_error:
        return f"Static check failed: {cand.static_error}"
    r = cand.report
    if r.sandbox_error:
        return f"Running the code failed: {r.sandbox_error}"
    if r.load_error:
        return f"Loading the code raised: {r.load_error}"
    lines = []
    for i, x in enumerate(r.results):
        if not x["ok"]:
            kind = "problem example" if i < cand.n_visible else "generated test"
            lines.append(f"- [{kind}] `{x['test']}` -> {x['error']}")
    return "Failed tests:\n" + "\n".join(lines[:8])


TEST_CAVEAT = (
    "Tests marked [problem example] come from the problem statement and are authoritative. "
    "Tests marked [generated test] were written by a Tester from the specification and may "
    "occasionally be wrong; if you are sure a generated test contradicts the specification, "
    "keep the behaviour the specification requires."
)


def coder_repair_leaf(task: Task, dataset: str, node, deps_code: str, cand) -> str:
    deps = (f"Helpers it may call (already tested, do not redefine):\n```python\n{deps_code.strip()}\n```\n\n"
            if deps_code.strip() else "")
    return f"""Overall problem (for context only):
{problem_text(task, dataset)}

Helper function:
- Signature: `{node.signature}`
- Specification: {node.description}

{deps}Current implementation:
```python
{cand.code.strip()}
```

{_failures_text(cand)}

{TEST_CAVEAT}
Find the bug and write the corrected complete `{node.name}` function in one ```python block.
{CODE_RULES}"""


def coder_repair_root(task: Task, dataset: str, cand) -> str:
    return f"""Problem:
{problem_text(task, dataset)}

Current program:
```python
{cand.code.strip()}
```

{_failures_text(cand)}

{TEST_CAVEAT}
The bug may be in `{task.entry_point}` or in any helper. Write the corrected complete program
(all functions it needs, including `{task.entry_point}`) in one ```python block.
{CODE_RULES}"""


# ------------------------------------------------------------------ Verifier

VERIFIER_SYSTEM = (
    "You are the Tester in a multi-agent programming team. You write unit tests from the "
    "specification alone; you never see the implementation, so your tests are independent of it."
)


def verifier_tests(spec: str, signature_or_name: str, n: int) -> str:
    return f"""Write {n} unit tests for this function.

Function: `{signature_or_name}`
Specification:
{spec}

Write each test as one plain `assert` statement on one line, for example `assert f(2, 3) == 5`.
Cover typical inputs and edge cases (empty inputs, boundaries, single elements, ties).
Work out every expected value carefully from the specification; only include cases whose expected
value you are sure of. Do not import or define anything and do not use a test framework.
Put all asserts in one ```python block."""
