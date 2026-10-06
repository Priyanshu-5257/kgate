"""Small helpers shared by the CLI. Nothing here talks to the network."""

from __future__ import annotations

import json
import re
from typing import Any


TUNNEL_HOST = re.compile(r"https://[A-Za-z0-9-]+\.trycloudflare\.com")
_STATUS_RE = re.compile(r'has status "([^"]+)"')
_FAIL_RE = re.compile(r'Failure message: "([^"]*)"')
_PUSH_VERSION_RE = re.compile(r"Kernel version (\d+) successfully pushed")
_PUSH_URL_RE = re.compile(r"https://(?:www\.)?kaggle\.com/\S+")
_HOURS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def parse_hours(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = _HOURS_RE.search(str(value))
    if not match:
        return 0.0
    return float(match.group(1))


def normalize_kernel_status(status: str) -> str:
    """Turn KernelWorkerStatus.RUNNING into RUNNING."""
    text = status.strip().strip('"')
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper() or "UNKNOWN"


def parse_kernel_status(text: str) -> tuple[str, str]:
    """Return (status, failure_message) from `kaggle kernels status` output."""
    status_match = _STATUS_RE.search(text)
    fail_match = _FAIL_RE.search(text)
    raw = status_match.group(1) if status_match else "UNKNOWN"
    failure = fail_match.group(1) if fail_match else ""
    return normalize_kernel_status(raw), failure


def parse_push_result(text: str) -> dict[str, Any]:
    version = _PUSH_VERSION_RE.search(text)
    url = _PUSH_URL_RE.search(text)
    error = ""
    for line in text.splitlines():
        if "Kernel push error" in line:
            error = line.split("Kernel push error:", 1)[-1].strip() or line.strip()
    return {
        "ok": bool(version) and not error,
        "version": int(version.group(1)) if version else None,
        "url": url.group(0).rstrip(").,") if url else "",
        "error": error,
    }


def parse_marker(logs: str, kind: str, run_id: str) -> dict[str, Any] | None:
    """Find a KGATE_TUNNEL or KGATE_ENGINE line for this run."""
    prefix = f"KGATE_{kind} "
    for line in logs.splitlines():
        stripped = line.strip()
        # Kaggle sometimes prefixes stream names. Accept a trailing JSON object.
        idx = stripped.find(prefix)
        if idx < 0:
            continue
        raw = stripped[idx + len(prefix) :]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("run") == run_id:
            return payload
    return None


def parse_username(config_view: str) -> str:
    for line in config_view.splitlines():
        if "username:" in line.lower():
            return line.split(":", 1)[1].strip()
    return ""


def default_dtype(accelerator: str) -> str:
    name = accelerator.lower()
    if any(token in name for token in ("a100", "h100", "l4", "b200")):
        return "bfloat16"
    # T4 and P100 have no bfloat16.
    return "half"


def plan_hours(requested: float, remaining: float | None) -> tuple[float, str]:
    """Fit the session timeout to the GPU hours still on the account.

    Returns (hours, warning). A remaining value of None means quota could not
    be read; the request is left unchanged and the caller should say so.
    """
    if remaining is None:
        return requested, "GPU quota could not be read. The session will still start."
    if remaining <= 0.05:
        raise ValueError("GPU quota on this account is used up.")
    if requested > remaining:
        usable = max(0.1, round(remaining - 0.05, 2))
        return (
            usable,
            f"Only {remaining:.2f}h of GPU quota is left, so this session is capped at {usable:.2f}h.",
        )
    return requested, ""


def redact(text: str, secrets: list[str]) -> str:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def public_session(session: dict[str, Any]) -> dict[str, Any]:
    hidden = {"token", "hf_token"}
    return {key: value for key, value in session.items() if key not in hidden}
