"""Framework hyper-parameters (symbols follow 框架形式化定义_v1.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FrameworkConfig:
    # --- decomposition (v0: fixed, every problem is split once into helpers + entry function)
    max_subtasks: int = 5          # B_max
    depth: int = 1                 # fixed decomposition depth; dynamic depth (D_max, θ_dec) comes in W3
    # --- verification (v0: one fixed level for every node; adaptive L0–L3 comes in W4)
    leaf_level: int = 2            # 0 = none, 1 = static checks, 2 = static + tests in the sandbox
    root_level: int = 2
    n_generated_tests: int = 6     # Verifier-written tests per node (written from the spec, not the code)
    per_test_timeout: int = 3      # seconds per test inside the sandbox
    # --- repair
    max_repairs: int = 2           # R
    max_calls: int = 40            # budget B: hard cap on LLM calls per problem

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # W2 framework v0: fixed decomposition + fixed L2 verification on every node
    "hmas-v0": FrameworkConfig(),
}
