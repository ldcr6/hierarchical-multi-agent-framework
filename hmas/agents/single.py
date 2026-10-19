"""Single-agent baselines.

DirectAgent: one LLM call, no planning, no verification ("Direct" row in the tables).
The prompt is EvalPlus's own chat-model prompt, so scores are comparable with the
EvalPlus leaderboard numbers for the same model.

CanonicalAgent: returns the reference solution; used only to check that the
evaluation pipeline scores 100% (never reported as a result).
"""
from __future__ import annotations

from ..bench.datasets import Task
from ..codeutil import extract_code
from ..llm import LLMClient, Usage
from .base import AgentResult

# Same as evalplus 0.3.1 codegen.py + provider/openai.py, for leaderboard comparability.
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
def task_prompt(task: Task) -> str:
    return f"{INSTRUCTION_PREFIX}\n```python\n{task.prompt.strip()}\n```"


class DirectAgent:
    name = "direct"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def solve(self, task: Task, dataset: str) -> AgentResult:
        usage = Usage()
        messages = [{"role": "user", "content": task_prompt(task)}]
        r = self.llm.chat(messages, usage=usage)
        code = extract_code(r.content)
        return AgentResult(solution=code, usage=usage,
                           trace=[{"step": "generate", "response": r.content}])


class CanonicalAgent:
    name = "canonical"

    def solve(self, task: Task, dataset: str) -> AgentResult:
        if dataset == "humaneval":
            sol = task.prompt + task.canonical_solution
        else:  # MBPP+ canonical solutions are complete programs
            sol = task.canonical_solution
        return AgentResult(solution=sol)
