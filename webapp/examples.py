"""Built-in demo problems and the scripted model replies used by offline demo mode.

Offline mode replaces the LLM with `ScriptedLLM`, which answers each Planner /
Coder / Verifier prompt from the script below, so the whole pipeline (planning,
coding, sandbox tests, repair) can be shown without network or API cost. The
first draft of `is_prime` is deliberately wrong so the repair loop is visible.
Everything else (prompt building, plan parsing, static checks, running the tests)
is the real framework code.
"""
from __future__ import annotations

import re
import time

from hmas.framework import prompts
from hmas.llm import LLMResponse, Usage

FENCE = "```"


def _py(code: str, confidence: float | None = None) -> str:
    out = f"{FENCE}python\n{code.strip()}\n{FENCE}"
    if confidence is not None:
        out += f"\nCONFIDENCE: {confidence}"
    return out


def _json(text: str) -> str:
    return f"{FENCE}json\n{text.strip()}\n{FENCE}"


EXAMPLES = [
    {
        "id": "prime_fib",
        "title": "HumanEval/39 prime_fib：第 n 个既是斐波那契数又是素数的数",
        "style": "humaneval",
        "prompt": '''

def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
''',
        "entry_point": "prime_fib",
        "acceptance": "assert prime_fib(6) == 233\nassert prime_fib(7) == 1597\nassert prime_fib(10) == 433494437",
        "note": "离线脚本里 is_prime 的第一版故意漏掉 k < 2 的情况，用来演示“验证失败 → 修复”。",
        "script": {
            "plan": _json('''
{
  "complexity": 3,
  "subtasks": [
    {"name": "is_prime", "signature": "def is_prime(k: int) -> bool",
     "description": "Return True if k is a prime number, False otherwise. Numbers below 2 (0, 1, negatives) are not prime.",
     "depends_on": [], "complexity": 2},
    {"name": "fibonacci", "signature": "def fibonacci(i: int) -> int",
     "description": "Return the i-th Fibonacci number with fibonacci(0) == 0 and fibonacci(1) == 1.",
     "depends_on": [], "complexity": 1}
  ],
  "composition": "Walk i = 0, 1, 2, ...; count the values fibonacci(i) for which is_prime is True and return the n-th one."
}'''),
            "code": {
                "is_prime": [
                    _py('''
def is_prime(k: int) -> bool:
    for d in range(2, int(k ** 0.5) + 1):
        if k % d == 0:
            return False
    return True
''', 0.9),
                    _py('''
def is_prime(k: int) -> bool:
    if k < 2:
        return False
    for d in range(2, int(k ** 0.5) + 1):
        if k % d == 0:
            return False
    return True
'''),
                ],
                "fibonacci": [_py('''
def fibonacci(i: int) -> int:
    a, b = 0, 1
    for _ in range(i):
        a, b = b, a + b
    return a
''', 0.95)],
                "prime_fib": [_py('''
def prime_fib(n: int):
    count = 0
    i = 0
    while True:
        f = fibonacci(i)
        if is_prime(f):
            count += 1
            if count == n:
                return f
        i += 1
''', 0.9)],
            },
            "tests": {
                "is_prime": _py('''
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(1) == False
assert is_prime(0) == False
assert is_prime(97) == True
'''),
                "fibonacci": _py('''
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
assert fibonacci(20) == 6765
'''),
                "prime_fib": _py('''
assert prime_fib(6) == 233
assert prime_fib(7) == 1597
assert prime_fib(8) == 28657
assert prime_fib(9) == 514229
'''),
            },
        },
    },
    {
        "id": "top_k_words",
        "title": "自然语言需求：统计英文文本中出现次数最多的 k 个单词",
        "style": "mbpp",
        "prompt": '''编写函数 top_k_words(text, k)：统计一段英文文本中出现次数最多的 k 个单词，按出现次数从高到低返回单词列表。单词不区分大小写（统一转为小写），只由英文字母组成，标点、空白和数字都视为分隔符；次数相同时按字母序排列；k 大于不同单词的个数时返回全部单词。
assert top_k_words("The cat and the hat.", 1) == ["the"]
assert top_k_words("b a b a c", 2) == ["a", "b"]''',
        "entry_point": "top_k_words",
        "acceptance": 'assert top_k_words("", 3) == []\nassert top_k_words("Hi! hi? HI... bye", 5) == ["hi", "bye"]',
        "note": "三个子任务一次通过，演示正常流程。",
        "script": {
            "plan": _json('''
{
  "complexity": 2,
  "subtasks": [
    {"name": "tokenize", "signature": "def tokenize(text: str) -> list",
     "description": "Split text into lowercase words made only of English letters; every other character is a separator. Return [] for text without letters.",
     "depends_on": [], "complexity": 1},
    {"name": "count_words", "signature": "def count_words(words: list) -> dict",
     "description": "Return a dict mapping each word to the number of times it occurs in words.",
     "depends_on": [], "complexity": 1},
    {"name": "rank_words", "signature": "def rank_words(counts: dict, k: int) -> list",
     "description": "Return the k words with the highest counts, ordered by count descending and alphabetically for ties; return all words if k exceeds their number.",
     "depends_on": [], "complexity": 2}
  ],
  "composition": "top_k_words returns rank_words(count_words(tokenize(text)), k)."
}'''),
            "code": {
                "tokenize": [_py('''
import re


def tokenize(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())
''', 0.95)],
                "count_words": [_py('''
def count_words(words: list) -> dict:
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts
''', 0.97)],
                "rank_words": [_py('''
def rank_words(counts: dict, k: int) -> list:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ordered[:k]]
''', 0.9)],
                "top_k_words": [_py('''
def top_k_words(text, k):
    return rank_words(count_words(tokenize(text)), k)
''', 0.95)],
            },
            "tests": {
                "tokenize": _py('''
assert tokenize("Hello, world!") == ["hello", "world"]
assert tokenize("") == []
assert tokenize("a1b") == ["a", "b"]
assert tokenize("It's OK") == ["it", "s", "ok"]
'''),
                "count_words": _py('''
assert count_words(["a", "b", "a"]) == {"a": 2, "b": 1}
assert count_words([]) == {}
assert count_words(["x"]) == {"x": 1}
'''),
                "rank_words": _py('''
assert rank_words({"a": 2, "b": 2, "c": 1}, 2) == ["a", "b"]
assert rank_words({"x": 1}, 5) == ["x"]
assert rank_words({}, 3) == []
assert rank_words({"b": 1, "a": 3}, 1) == ["a"]
'''),
                "top_k_words": _py('''
assert top_k_words("", 3) == []
assert top_k_words("Go go GO stop", 2) == ["go", "stop"]
assert top_k_words("x y z", 10) == ["x", "y", "z"]
'''),
            },
        },
    },
]

BY_ID = {e["id"]: e for e in EXAMPLES}


def public_examples() -> list[dict]:
    keys = ("id", "title", "style", "prompt", "entry_point", "acceptance", "note")
    return [{k: e[k] for k in keys} for e in EXAMPLES]


class ScriptedLLM:
    """Drop-in for LLMClient.chat that answers from an example's script (offline demo)."""

    def __init__(self, example: dict, delay: float = 0.8):
        self.script = example["script"]
        self.delay = delay
        self.served: dict[str, int] = {}
        self.total = Usage()

    def _reply(self, system: str, user: str) -> str:
        if system == prompts.PLANNER_SYSTEM:
            return self.script["plan"]
        if system == prompts.VERIFIER_SYSTEM:
            m = re.search(r"Function: `(?:def )?(\w+)", user)
            return self.script["tests"].get(m.group(1) if m else "", _py("assert True"))
        # Coder: which function is it about?
        m = (re.search(r"Signature: `def (\w+)", user)
             or re.search(r"corrected complete `(\w+)`", user)
             or re.search(r"entry function `(\w+)`", user)
             or re.search(r"the function `(\w+)`", user)
             or re.search(r"including `(\w+)`\)", user))
        name = m.group(1) if m else ""
        versions = self.script["code"].get(name)
        if not versions:
            return "(no scripted reply)"
        i = self.served.get(name, 0)
        self.served[name] = i + 1
        return versions[min(i, len(versions) - 1)]

    def chat(self, messages, *, temperature=None, max_tokens=None, sample_index=0, usage=None):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        time.sleep(self.delay)
        text = self._reply(system, user)
        resp = LLMResponse(text, prompt_tokens=(len(system) + len(user)) // 4,
                           completion_tokens=len(text) // 4, latency=self.delay)
        self.total.add(resp)
        if usage is not None:
            usage.add(resp)
        return resp
