"""Authenticated calls from the laptop to a running session gateway."""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit


class ControlError(RuntimeError):
    pass


def _opener(base_url: str) -> tuple[str, float]:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"}:
        raise ControlError(f"unsupported tunnel URL {base_url}")
    return base_url.rstrip("/"), 30


def request_json(
    base_url: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    root, _default = _opener(base_url)
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(root + path, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ControlError(f"{method} {path} failed ({exc.code}): {detail[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
        raise ControlError(f"{method} {path} failed: {exc}") from exc
    if not raw:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ControlError("session gateway returned a non-object JSON body")
    return loaded


def health(base_url: str, token: str, timeout: float = 15) -> dict[str, Any]:
    return request_json(base_url, token, "GET", "/kgate/health", timeout=timeout)


def stop_session(base_url: str, token: str) -> dict[str, Any]:
    return request_json(base_url, token, "POST", "/kgate/stop", {}, timeout=20)


def exec_command(base_url: str, token: str, command: str, timeout: float = 30) -> dict[str, Any]:
    return request_json(
        base_url,
        token,
        "POST",
        "/kgate/exec",
        {"cmd": command, "timeout": timeout},
        timeout=timeout + 10,
    )
