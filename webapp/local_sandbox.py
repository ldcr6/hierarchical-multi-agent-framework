"""Fallback test runner for development machines without Docker.

It runs the test harness in a plain subprocess, with NO isolation, so it is only
used in offline demo mode, where every program comes from our own scripts
(webapp/examples.py), never from a model. Real model output always goes to the
Docker sandbox. The harness needs SIGALRM, so this only works on Linux / macOS.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from hmas.sandbox import ExecResult


class LocalSandbox:
    @staticmethod
    def available() -> bool:
        return hasattr(signal, "SIGALRM")

    def run_python(self, code, *, stdin_files=None, timeout=None, args=()):
        with tempfile.TemporaryDirectory(prefix="hmas-web-") as d:
            Path(d, "main.py").write_text(code, encoding="utf-8")
            for name, content in (stdin_files or {}).items():
                Path(d, name).write_text(content, encoding="utf-8")
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, "-I", "main.py", *args], cwd=d, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=timeout)
                return ExecResult(p.returncode, p.stdout, p.stderr, False, time.time() - t0)
            except subprocess.TimeoutExpired:
                return ExecResult(-1, "", "", True, time.time() - t0)
