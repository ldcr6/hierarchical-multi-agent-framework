"""LangGraph orchestration of the Planner -> Coder -> Verifier loop (framework v0).

    plan -> code -> verify --(passed / out of repairs / out of budget)--> advance --(more nodes)--> code
                       ^  \\--(failed)--> repair --/                         \\--(done)--> finalize

Nodes are solved bottom-up: helpers in dependency order, then the entry function
(the root), which is composed from the verified helpers. Each repair works on the
latest version and stops early when it reproduces an earlier one. Each node keeps
every candidate it produced; the best-scoring one (earliest on ties) is kept.

v0 = fixed decomposition (one level of helpers for every problem) and fixed
verification (the same level for every node). The dynamic depth (W3) and the
risk-based L0–L3 choice (W4) replace `plan` and `node.level` respectively.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..bench.datasets import Task
from ..agents.base import AgentResult
from ..llm import LLMClient
from ..sandbox import DockerSandbox
from .agents import Coder, Planner, Run, Verifier
from .config import FrameworkConfig


class State(TypedDict):
    run: Run
    cursor: int        # index into run.nodes of the node being solved
    repairs: int       # repairs done on the current node


def build_graph(planner: Planner, coder: Coder, verifier: Verifier):
    def plan(s: State):
        planner.decompose(s["run"])
        return {"cursor": 0, "repairs": 0}

    def code(s: State):
        run = s["run"]
        node = run.nodes[s["cursor"]]
        node.candidates.append(coder.implement(run, node))
        return {"repairs": 0}

    def verify(s: State):
        run = s["run"]
        node = run.nodes[s["cursor"]]
        verifier.verify(run, node, node.candidates[-1])
        return {}

    def route_after_verify(s: State) -> str:
        run = s["run"]
        node = run.nodes[s["cursor"]]
        last = node.candidates[-1]
        if last.passed or s["repairs"] >= run.cfg.max_repairs or run.over_budget():
            return "advance"
        # a repair that reproduced an earlier version would get the same prompt again
        if any(c.code == last.code for c in node.candidates[:-1]):
            return "advance"
        return "repair"

    def repair(s: State):
        run = s["run"]
        node = run.nodes[s["cursor"]]
        # repair the latest version with its own failures (self-debugging); the final pick
        # is still the best-scoring version, so a worse repair is never kept
        node.candidates.append(coder.repair(run, node, node.candidates[-1]))
        return {"repairs": s["repairs"] + 1}

    def advance(s: State):
        run = s["run"]
        node = run.nodes[s["cursor"]]
        node.final = node.best()
        return {"cursor": s["cursor"] + 1}

    def route_after_advance(s: State) -> str:
        return "code" if s["cursor"] < len(s["run"].nodes) else END

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("code", code), ("verify", verify), ("repair", repair), ("advance", advance)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"advance": "advance", "repair": "repair"})
    g.add_edge("repair", "verify")
    g.add_conditional_edges("advance", route_after_advance, {"code": "code", END: END})
    return g.compile()


class HierarchicalAgent:
    """Planner + Coder + Verifier framework; implements the common Agent interface."""

    def __init__(self, llm: LLMClient, cfg: FrameworkConfig | None = None,
                 sandbox: DockerSandbox | None = None, name: str = "hmas-v0"):
        self.name = name
        self.cfg = cfg or FrameworkConfig()
        self.sandbox = sandbox or DockerSandbox(timeout=60)
        self.graph = build_graph(Planner(llm), Coder(llm), Verifier(llm, self.sandbox))

    def solve(self, task: Task, dataset: str) -> AgentResult:
        run = Run(task=task, dataset=dataset, cfg=self.cfg)
        self.graph.invoke({"run": run, "cursor": 0, "repairs": 0}, config={"recursion_limit": 500})
        solution = run.root.final.code
        extra = {
            "usage_by_role": {r: u.as_dict() for r, u in run.usage.items()},
            "n_helpers": len(run.helpers),
            "plan_problems": run.plan_problems,
            "nodes": [n.summary() for n in run.nodes],
        }
        return AgentResult(solution=solution, usage=run.total_usage(), trace=run.trace, extra=extra)
