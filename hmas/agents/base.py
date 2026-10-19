"""Common agent interface: every method (baselines and our framework) maps a Task to a solution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..bench.datasets import Task
from ..llm import Usage


@dataclass
class AgentResult:
    solution: str                      # full program text (EvalPlus "solution" field)
    usage: Usage = field(default_factory=Usage)
    trace: list[dict] = field(default_factory=list)  # per-step log for analysis / case studies


class Agent(Protocol):
    name: str

    def solve(self, task: Task, dataset: str) -> AgentResult: ...
