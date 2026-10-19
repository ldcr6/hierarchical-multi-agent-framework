"""Helpers for pulling code out of model responses."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*(?:python|py|Python|python3)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Return the longest fenced code block, or the whole text if there is none.

    Final cleanup (dropping test calls, keeping only what the entry point needs)
    is done by EvalPlus's own sanitizer inside the sandbox, so results stay
    comparable with the EvalPlus leaderboard.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    # an unterminated fence (response cut off by max_tokens)
    m = re.search(r"```[ \t]*(?:python|py)?[ \t]*\n(.*)", text, re.S)
    if m:
        return m.group(1)
    return text
