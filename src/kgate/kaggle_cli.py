"""Talk to the Kaggle CLI with one account's token in the environment.

The Kaggle library keeps credentials on the class, so a long-lived process
that switches accounts is safer calling the CLI in a subprocess. The token
is passed as KAGGLE_API_TOKEN and is never written into log files by this
module.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import time
from typing import Any, BinaryIO

from kgate.util import parse_kernel_status, parse_push_result, parse_username, redact


class KaggleError(RuntimeError):
    pass


def kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if not found:
        raise KaggleError(
            "The kaggle command is not on PATH. Install it in this environment "
            "(pip install kaggle) and run `kaggle auth login`, or activate the "
            "virtualenv that already has it."
        )
    return found


def run_kaggle(token: str, args: list[str], timeout: float = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    try:
        completed = subprocess.run(
            [kaggle_bin(), *args],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise KaggleError(f"kaggle {' '.join(args)} timed out after {timeout:.0f}s") from exc
    return completed


def _output(completed: subprocess.CompletedProcess[str], token: str) -> str:
    return redact((completed.stdout or "") + (completed.stderr or ""), [token])


def whoami(token: str) -> str:
    completed = run_kaggle(token, ["config", "view"], timeout=60)
    username = parse_username(_output(completed, token))
    if completed.returncode != 0 or not username:
        detail = _output(completed, token).strip() or f"exit {completed.returncode}"
        raise KaggleError(f"Could not authenticate this token. {detail}")
    return username


def quota(token: str) -> list[dict[str, Any]]:
    completed = run_kaggle(token, ["quota", "--format", "json"], timeout=60)
    text = _output(completed, token)
    if completed.returncode != 0:
        raise KaggleError(text.strip() or "kaggle quota failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise KaggleError(f"Could not parse quota JSON: {text[:500]}") from exc
    if not isinstance(payload, list):
        raise KaggleError("Unexpected quota payload")
    return payload


def gpu_remaining_hours(rows: list[dict[str, Any]]) -> float | None:
    from kgate.util import parse_hours

    for row in rows:
        if str(row.get("resource", "")).upper() == "GPU":
            if "remaining" in row:
                return parse_hours(row.get("remaining"))
    return None


def kernel_status(token: str, kernel: str) -> tuple[str, str]:
    completed = run_kaggle(token, ["kernels", "status", kernel], timeout=60)
    text = _output(completed, token)
    if completed.returncode != 0 and "has status" not in text:
        return "ABSENT", text.strip()
    return parse_kernel_status(text)


def collect_stream(stream: BinaryIO, idle_s: float, max_s: float) -> str:
    """Read a pipe until it closes, goes idle, or max_s elapses.

    Kaggle's one-shot log call is empty while a notebook is running. The live
    stream replays the whole log, then waits. Idle detection returns that replay
    without sitting on the stream forever.
    """
    fd = stream.fileno()
    chunks: list[bytes] = []
    deadline = time.time() + max_s
    last = time.time()
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            if chunks and time.time() - last >= idle_s:
                break
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
        last = time.time()
    return b"".join(chunks).decode("utf-8", errors="replace")


def _logs_process(token: str, kernel: str) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    return subprocess.Popen(
        [kaggle_bin(), "kernels", "logs", "-f", kernel],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        bufsize=0,
    )


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def kernel_logs(token: str, kernel: str, idle_s: float = 1.2, max_s: float = 20) -> str:
    """Return the live notebook log. Empty only if the stream itself is empty."""
    proc = _logs_process(token, kernel)
    try:
        assert proc.stdout is not None
        text = collect_stream(proc.stdout, idle_s, max_s)
    finally:
        _stop_process(proc)
    return redact(text, [token])


def follow_kernel_logs(token: str, kernel: str):
    """Yield live log chunks until the stream ends or the caller stops iterating."""
    proc = _logs_process(token, kernel)
    try:
        assert proc.stdout is not None
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if proc.poll() is not None and not ready:
                break
            if not ready:
                continue
            data = os.read(proc.stdout.fileno(), 65536)
            if not data:
                break
            yield redact(data.decode("utf-8", errors="replace"), [token])
    finally:
        _stop_process(proc)


def push_kernel(
    token: str,
    folder: str,
    timeout_seconds: int,
    accelerator: str,
) -> dict[str, Any]:
    completed = run_kaggle(
        token,
        [
            "kernels",
            "push",
            "-p",
            folder,
            "-t",
            str(timeout_seconds),
            "--accelerator",
            accelerator,
        ],
        timeout=180,
    )
    text = _output(completed, token)
    parsed = parse_push_result(text)
    parsed["output"] = text.strip()
    if completed.returncode != 0 or not parsed["ok"]:
        detail = parsed["error"] or text.strip() or f"exit {completed.returncode}"
        raise KaggleError(detail)
    return parsed
