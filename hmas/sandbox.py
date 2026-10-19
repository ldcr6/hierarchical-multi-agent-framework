"""Docker sandbox for running untrusted, model-generated code.

Each run starts a fresh container with no network, a read-only root filesystem,
and CPU / memory / process limits. The same image also hosts the official
EvalPlus evaluator, so scoring never runs generated code on the host.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "hmas-sandbox:0.2"
DOCKERFILE = Path(__file__).resolve().parent.parent / "docker" / "sandbox.Dockerfile"


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class DockerSandbox:
    def __init__(self, image: str = DEFAULT_IMAGE, cpus: float = 1.0, memory: str = "1g",
                 pids: int = 256, timeout: float = 10.0):
        self.image, self.cpus, self.memory, self.pids, self.timeout = image, cpus, memory, pids, timeout

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    def image_exists(self) -> bool:
        return subprocess.run(["docker", "image", "inspect", self.image], capture_output=True).returncode == 0

    def run_cmd(self, cmd: list[str], *, mounts: list[tuple[str, str, str]] = (),
                env: dict[str, str] | None = None, timeout: float | None = None,
                workdir: str = "/work", cpus: float | None = None, memory: str | None = None,
                read_only: bool = True, user: str | None = None) -> ExecResult:
        """Run `cmd` in a fresh container. mounts: (host_path, container_path, "ro"|"rw")."""
        timeout = self.timeout if timeout is None else timeout
        name = f"hmas-{uuid.uuid4().hex[:12]}"
        args = ["docker", "run", "--rm", "--name", name, "--network", "none",
                "--cpus", str(cpus or self.cpus), "--memory", memory or self.memory,
                "--pids-limit", str(self.pids), "--security-opt", "no-new-privileges",
                "--tmpfs", "/tmp:rw,size=256m", "-w", workdir]
        if read_only:
            args.append("--read-only")
        if user:
            args += ["--user", user]
        for host, cont, mode in mounts:
            args += ["-v", f"{Path(host).resolve()}:{cont}:{mode}"]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [self.image, *cmd]

        t0 = time.time()
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return ExecResult(p.returncode, p.stdout, p.stderr, False, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True)
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return ExecResult(-1, out, err, True, time.time() - t0)

    def run_python(self, code: str, *, stdin_files: dict[str, str] | None = None,
                   timeout: float | None = None) -> ExecResult:
        """Execute a Python script (plus optional extra files) inside the sandbox."""
        with tempfile.TemporaryDirectory(prefix="hmas-") as d:
            Path(d, "main.py").write_text(code, encoding="utf-8")
            for fname, content in (stdin_files or {}).items():
                Path(d, fname).write_text(content, encoding="utf-8")
            Path(d).chmod(0o755)
            for f in Path(d).iterdir():
                f.chmod(0o644)
            return self.run_cmd(["python", "main.py"], mounts=[(d, "/work", "ro")], timeout=timeout)
