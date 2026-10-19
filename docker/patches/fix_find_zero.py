"""Backport an upstream EvalPlus fix into the PyPI release 0.3.1.

In evalplus 0.3.1 the special oracle for HumanEval/32 (find_zero) `continue`s
without recording the test as passed, so every solution, including the
reference one, is scored as failing. Upstream master records the pass
(evalplus/eval/__init__.py). This script applies the same three lines.
"""
import pathlib

import evalplus.eval as E

path = pathlib.Path(E.__file__)
src = path.read_text()
old = (
    "                            assert abs(_poly(*inp, out)) <= atol\n"
    "                            continue\n"
)
new = (
    "                            assert abs(_poly(*inp, out)) <= atol\n"
    "                            details[i] = True\n"
    "                            progress.value += 1\n"
    "                            continue\n"
)
if new in src:
    print("already patched")
else:
    assert old in src, "unexpected evalplus source; check the version"
    path.write_text(src.replace(old, new))
    print("patched", path)
