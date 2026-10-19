from hmas.agents.single import CanonicalAgent, DirectAgent, task_prompt
from hmas.bench.datasets import Task
from hmas.llm import LLMResponse

TASK = Task("HumanEval/0", 'def add(a, b):\n    """Add two numbers."""\n', "add", "    return a + b\n")


class FakeLLM:
    def chat(self, messages, usage=None, **kw):
        r = LLMResponse("Sure:\n```python\ndef add(a, b):\n    return a + b\n```", 30, 12, 0.0)
        if usage is not None:
            usage.add(r)
        return r


def test_direct_agent():
    res = DirectAgent(FakeLLM()).solve(TASK, "humaneval")
    assert res.solution == "def add(a, b):\n    return a + b\n"
    assert res.usage.calls == 1 and res.usage.completion_tokens == 12


def test_prompt_matches_evalplus_format():
    p = task_prompt(TASK)
    assert p.startswith("Please provide a self-contained Python script")
    assert "```python\ndef add(a, b):" in p


def test_canonical_agent():
    assert CanonicalAgent().solve(TASK, "humaneval").solution == TASK.prompt + TASK.canonical_solution
