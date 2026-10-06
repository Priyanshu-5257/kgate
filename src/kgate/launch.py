"""Build a private Kaggle kernel and point this laptop at it."""

from __future__ import annotations

import json
import os
import secrets
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kgate import kaggle_cli
from kgate.control import ControlError, health, stop_session
from kgate.proxy import pick_port, spawn, stop_pid
from kgate.remote_agent import engine_argv
from kgate.store import Store
from kgate.util import default_dtype, parse_marker, plan_hours, public_session

AGENT_PATH = Path(__file__).resolve().parent / "remote_agent.py"
TERMINAL_STATUSES = {"ERROR", "COMPLETE", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"}


def render_session_source(cfg: dict[str, Any]) -> str:
    agent = AGENT_PATH.read_text(encoding="utf-8")
    header = "import json\nCONFIG = json.loads(" + json.dumps(json.dumps(cfg)) + ")\n\n"
    return header + agent


def kernel_metadata(username: str, slug: str, accelerator: str) -> dict[str, Any]:
    use_tpu = accelerator.lower().startswith("tpu")
    return {
        "id": f"{username}/{slug}",
        "title": "kgate session",
        "code_file": "session.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not use_tpu,
        "enable_tpu": use_tpu,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def build_config(
    *,
    run_id: str,
    token: str,
    engine: str,
    model: str,
    accelerator: str,
    hours: float,
    engine_args: list[str],
    max_model_len: int,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    quantization: str,
    hf_token: str,
    dtype: str,
    studio_password: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "token": token,
        "engine": engine,
        "model": model,
        "engine_port": 8000,
        "gate_port": 8788,
        "studio_port": 8888,
        "studio_password": studio_password,
        "dtype": dtype or default_dtype(accelerator),
        "max_model_len": max_model_len,
        "tensor_parallel": tensor_parallel,
        "gpu_memory_utilization": gpu_memory_utilization,
        "quantization": quantization,
        "engine_args": engine_args,
        "hf_token": hf_token,
        "trust_remote_code": True,
        "die_after_s": int(hours * 3600),
        "engine_timeout_s": 5400 if engine == "unsloth" else 2400,
    }


def write_kernel_dir(folder: Path, username: str, slug: str, accelerator: str, cfg: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    script = folder / "session.py"
    script.write_text(render_session_source(cfg), encoding="utf-8")
    os.chmod(script, 0o600)
    meta = folder / "kernel-metadata.json"
    meta.write_text(json.dumps(kernel_metadata(username, slug, accelerator), indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fresh_lines(previous: str, current: str) -> str:
    if not current or current == previous:
        return ""
    if previous and current.startswith(previous):
        return current[len(previous) :]
    if not previous:
        return current
    return ""


def wait_for_markers(
    token: str,
    kernel: str,
    run_id: str,
    timeout_s: float,
    want_engine: bool,
    on_line: Callable[[str], None],
    on_tunnel: Callable[[str], None],
) -> None:
    deadline = time.time() + timeout_s
    seen_logs = ""
    boot_seen = False
    tunnel_seen = False
    last_status = ""
    while time.time() < deadline:
        try:
            status, failure = kaggle_cli.kernel_status(token, kernel)
        except kaggle_cli.KaggleError as exc:
            status, failure = "UNKNOWN", str(exc)
        if status != last_status:
            on_line(f"[kgate] kernel status {status}\n")
            last_status = status
        try:
            logs = kaggle_cli.kernel_logs(token, kernel)
        except kaggle_cli.KaggleError:
            logs = ""
        boot = parse_marker(logs, "BOOT", run_id)
        if boot and not boot_seen:
            boot_seen = True
            seen_logs = ""
        if boot_seen:
            chunk = _fresh_lines(seen_logs, logs)
            if chunk:
                on_line(chunk if chunk.endswith("\n") else chunk + "\n")
                seen_logs = logs
        tunnel = parse_marker(logs, "TUNNEL", run_id)
        if tunnel and tunnel.get("error"):
            raise RuntimeError(str(tunnel["error"]))
        if tunnel and tunnel.get("url") and not tunnel_seen:
            tunnel_seen = True
            on_tunnel(str(tunnel["url"]))
            if not want_engine:
                return
        engine = parse_marker(logs, "ENGINE", run_id) if want_engine else None
        if engine:
            if engine.get("ready"):
                return
            if engine.get("error"):
                raise RuntimeError(str(engine["error"]))
        if boot_seen and status in TERMINAL_STATUSES:
            raise RuntimeError(failure or f"kernel ended with status {status}")
        time.sleep(5)
    if not boot_seen:
        if last_status in {"RUNNING", "QUEUED"}:
            raise TimeoutError(
                f"The notebook status is {last_status}, but its live log never showed this run. "
                "Open `kgate dash` for the log, or run `kgate logs --follow`."
            )
        raise TimeoutError(
            "The notebook did not start before the wait expired. "
            "It may still be queued. Open `kgate dash`, or run `kgate logs --follow`."
        )
    if not tunnel_seen:
        raise TimeoutError("The kernel started, but the tunnel URL never appeared. Run `kgate logs`.")
    raise TimeoutError("The tunnel is up, but the inference engine did not become ready. Run `kgate logs`.")


def _endpoint_path(store: Store, session_id: str) -> Path:
    directory = store.home / "endpoints"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory / f"{session_id}.json"


def ensure_proxy(store: Store, session: dict[str, Any]) -> dict[str, Any]:
    port = int(session.get("local_port") or 0)
    pid = int(session.get("proxy_pid") or 0)
    if port and pid and _proxy_alive(pid):
        return session
    preferred = port or int(store.config().get("local_port") or 8000)
    port = pick_port(preferred)
    path = _endpoint_path(store, session["id"])
    path.write_text(
        json.dumps({"tunnel_url": session["tunnel_url"], "token": session["token"]}) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    log_path = store.log_dir / f"proxy-{session['id']}.log"
    pid = spawn(port, str(path), str(log_path))
    if not _wait_port(port, 8):
        stop_pid(pid)
        raise RuntimeError(f"local proxy did not bind 127.0.0.1:{port}")
    session["local_port"] = port
    session["proxy_pid"] = pid
    store.upsert_session(session)
    return session


def _proxy_alive(pid: int) -> bool:
    proc_path = f"/proc/{pid}/cmdline"
    if not os.path.exists(proc_path):
        return False
    with open(proc_path, "rb") as handle:
        cmdline = handle.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return "kgate.proxy" in cmdline or "kgate/proxy" in cmdline


def _wait_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _mark_stopped(store: Store, session: dict[str, Any], error: str = "") -> None:
    stop_pid(int(session.get("proxy_pid") or 0))
    session["proxy_pid"] = 0
    session["status"] = "stopped"
    session["stopped_at"] = _now()
    if error:
        session["error"] = error
    store.upsert_session(session)


def probe(session: dict[str, Any]) -> dict[str, Any] | None:
    if not session.get("tunnel_url") or not session.get("token"):
        return None
    try:
        return health(str(session["tunnel_url"]), str(session["token"]), timeout=8)
    except ControlError:
        return None


def attach_existing(store: Store, session: dict[str, Any]) -> dict[str, Any] | None:
    live = probe(session)
    if not live:
        return None
    session["status"] = "ready" if live.get("engine_ready") or session.get("engine") == "none" else "starting"
    session["engine_ready"] = bool(live.get("engine_ready"))
    return ensure_proxy(store, session)


def stop_remote(store: Store, session: dict[str, Any], token: str) -> str:
    """Stop a session. Returns a note for the user. The local proxy is always stopped."""
    note = ""
    if session.get("tunnel_url") and session.get("token"):
        try:
            stop_session(str(session["tunnel_url"]), str(session["token"]))
            note = "Stop request sent to the Kaggle session."
        except ControlError as exc:
            note = (
                "Could not reach the session tunnel. Stop the notebook in the Kaggle UI if it is still running: "
                + str(session.get("kernel_url") or session.get("kernel") or "")
                + f" ({exc})"
            )
    else:
        note = "This session has no tunnel URL. Stop it from the Kaggle notebook page if it is still running."
    # Give the kernel a moment to exit, then record whatever status Kaggle reports.
    time.sleep(2)
    try:
        status, failure = kaggle_cli.kernel_status(token, str(session["kernel"]))
        if failure:
            note = f"{note} Kaggle status: {status}. {failure}".strip()
        else:
            note = f"{note} Kaggle status: {status}".strip()
    except kaggle_cli.KaggleError as exc:
        note = f"{note} Could not read kernel status: {exc}"
    _mark_stopped(store, session)
    return note


def status_blocks_new_launch(status: str, run_finished: bool) -> bool:
    """A QUEUED kernel is still starting. RUNNING blocks only while its log is live.

    After the notebook exits, Kaggle often leaves the status at RUNNING. That
    stale status must not refuse the next launch.
    """
    if status == "QUEUED":
        return True
    if status != "RUNNING":
        return False
    return not run_finished


def release_kernel(
    store: Store,
    token: str,
    kernel: str,
    session: dict[str, Any] | None,
) -> str:
    """Stop a kernel, or explain that Kaggle's RUNNING status is already stale."""
    status, failure = kaggle_cli.kernel_status(token, kernel)
    finished = status == "RUNNING" and kaggle_cli.kernel_run_finished(token, kernel)
    live = status_blocks_new_launch(status, finished)
    if live and session is not None:
        return stop_remote(store, session, token)
    if live:
        return (
            f"{kernel} is {status}. Stop it from https://www.kaggle.com/code/{kernel} "
            "before starting another. kgate has no tunnel URL for it."
        )
    if session is not None:
        _mark_stopped(store, session, failure or "")
    if status == "RUNNING" and finished:
        return (
            f"{kernel} is not running. Kaggle still reports RUNNING, but the notebook "
            "log has ended. `kgate up` can start a new session."
        )
    detail = f" {failure}" if failure else ""
    return f"Kaggle status: {status}.{detail} Nothing is running."


def require_model(engine: str, model: str) -> None:
    if engine not in {"none", "unsloth"} and not model:
        raise ValueError("pass --model, or use --engine none for a shell only, or --engine unsloth for the Studio UI")


def launch_session(
    store: Store,
    account: dict[str, Any],
    *,
    engine: str,
    model: str,
    accelerator: str,
    hours: float,
    engine_args: list[str],
    max_model_len: int,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    quantization: str,
    hf_token: str,
    dtype: str,
    local_port: int,
    wait: bool,
    slug: str,
) -> dict[str, Any]:
    require_model(engine, model)
    if engine != "none" and accelerator.lower().startswith("tpu"):
        raise ValueError("Unsloth, vLLM, SGLang, and Ollama need a GPU accelerator, not a TPU")
    token = store.read_token(account)
    username = kaggle_cli.whoami(token)
    if username != account.get("username"):
        account["username"] = username
        store.put_account(account)
    kernel = f"{username}/{slug}"
    for existing in store.active_sessions(account["name"]):
        if existing.get("kernel") != kernel:
            continue
        revived = attach_existing(store, existing)
        if revived:
            print(f"Session {revived['id']} is already up.", flush=True)
            return revived
    status, failure = kaggle_cli.kernel_status(token, kernel)
    finished = status == "RUNNING" and kaggle_cli.kernel_run_finished(token, kernel)
    if not status_blocks_new_launch(status, finished):
        for existing in store.active_sessions(account["name"]):
            if existing.get("kernel") == kernel:
                _mark_stopped(store, existing, failure or f"kernel status {status}")
    else:
        raise RuntimeError(
            f"{kernel} is already {status}. Stop it with `kgate down` or from "
            f"https://www.kaggle.com/code/{kernel} before starting another. {failure}".strip()
        )
    remaining = None
    quota_note = ""
    try:
        remaining = kaggle_cli.gpu_remaining_hours(kaggle_cli.quota(token))
    except kaggle_cli.KaggleError as exc:
        quota_note = str(exc)
    if accelerator.lower().startswith("tpu"):
        planned, warning = hours, ""
    else:
        planned, warning = plan_hours(hours, remaining)
    if warning:
        print(warning, flush=True)
    if quota_note:
        print(quota_note, flush=True)
    run_id = secrets.token_hex(8)
    gate_token = secrets.token_urlsafe(32)
    studio_password = secrets.token_urlsafe(12) if engine == "unsloth" else ""
    cfg = build_config(
        run_id=run_id,
        token=gate_token,
        engine=engine,
        model=model,
        accelerator=accelerator,
        hours=planned,
        engine_args=engine_args,
        max_model_len=max_model_len,
        tensor_parallel=tensor_parallel,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        hf_token=hf_token,
        dtype=dtype,
        studio_password=studio_password,
    )
    # engine_argv is validated locally so a bad engine fails before the push.
    if engine != "none":
        engine_argv(cfg, 1)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + run_id[:4]
    folder = store.kernel_dir / session_id
    write_kernel_dir(folder, username, slug, accelerator, cfg)
    print(
        f"Starting a private {accelerator} session on {username} for up to {planned:.2f}h. "
        "GPU quota ticks for the whole time the notebook is up.",
        flush=True,
    )
    pushed = kaggle_cli.push_kernel(token, str(folder), max(300, int(planned * 3600)), accelerator)
    session: dict[str, Any] = {
        "id": session_id,
        "run_id": run_id,
        "account": account["name"],
        "username": username,
        "kernel": kernel,
        "kernel_url": pushed.get("url") or f"https://www.kaggle.com/code/{kernel}",
        "version": pushed.get("version"),
        "accelerator": accelerator,
        "engine": engine,
        "model": model,
        "studio_password": studio_password,
        "ui_url": "",
        "dtype": cfg["dtype"],
        "hours": planned,
        "tunnel_url": "",
        "token": gate_token,
        "local_port": local_port,
        "proxy_pid": 0,
        "status": "starting",
        "engine_ready": engine == "none",
        "started_at": _now(),
        "error": "",
    }
    store.upsert_session(session)

    def on_tunnel(url: str) -> None:
        session["tunnel_url"] = url
        session["status"] = "starting"
        store.upsert_session(session)
        ensure_proxy(store, session)
        if session.get("engine") == "unsloth":
            print("Control tunnel is up. Unsloth Studio is still installing on the notebook.", flush=True)
            print("The browser link is printed when the UI is ready. `kgate dash` shows it too.", flush=True)
        else:
            base = f"http://127.0.0.1:{session['local_port']}/v1"
            print(f"Local API base URL: {base}", flush=True)
        print("The shell is available with `kgate attach`.", flush=True)

    if not wait:
        print(f"Pushed {session['kernel_url']}. Run `kgate logs` to watch it.", flush=True)
        return session
    try:
        wait_for_markers(
            token,
            kernel,
            run_id,
            timeout_s=80 * 60 if engine == "unsloth" else 45 * 60 if engine != "none" else 20 * 60,
            want_engine=engine != "none",
            on_line=lambda text: print(text, end="", flush=True),
            on_tunnel=on_tunnel,
        )
    except KeyboardInterrupt:
        print("\nStopped waiting. The Kaggle session is still running. `kgate down` stops it.", flush=True)
        return session
    session["status"] = "ready"
    session["engine_ready"] = True
    session["ready_at"] = _now()
    if engine == "unsloth":
        try:
            payload = parse_marker(kaggle_cli.kernel_logs(token, kernel), "ENGINE", run_id) or {}
        except kaggle_cli.KaggleError:
            payload = {}
        if payload.get("ui_url"):
            session["ui_url"] = str(payload["ui_url"])
    store.upsert_session(session)
    if engine == "unsloth":
        print(f"Unsloth Studio: {session.get('ui_url') or 'see `kgate logs`'}", flush=True)
        print(f"Password: {studio_password}", flush=True)
        print("Open that link and pick a model from the Hugging Face search.", flush=True)
    elif engine != "none":
        print(
            f"Model is ready. Point any OpenAI client at http://127.0.0.1:{session['local_port']}/v1 "
            f"with model {model!r}. The API key can be any non-empty string.",
            flush=True,
        )
    return session


def adopt_logs(store: Store, session: dict[str, Any], logs: str) -> dict[str, Any]:
    """If the live log already has the tunnel, record it and start the local proxy."""
    run_id = str(session.get("run_id") or "")
    if not run_id:
        return session
    before = (
        session.get("tunnel_url"),
        session.get("status"),
        session.get("engine_ready"),
        session.get("error"),
        session.get("engine_error"),
        session.get("ui_url"),
    )
    tunnel = parse_marker(logs, "TUNNEL", run_id)
    engine = parse_marker(logs, "ENGINE", run_id)
    if tunnel and tunnel.get("error"):
        session["status"] = "error"
        session["error"] = str(tunnel["error"])
    if tunnel and tunnel.get("url"):
        session["tunnel_url"] = str(tunnel["url"])
    if engine and engine.get("ui_url"):
        session["ui_url"] = str(engine["ui_url"])
    if engine and engine.get("ready"):
        session["status"] = "ready"
        session["engine_ready"] = True
        session["error"] = ""
    elif engine and engine.get("error"):
        session["engine_error"] = str(engine["error"])
    elif session.get("tunnel_url") and session.get("status") == "starting":
        session["status"] = "ready" if session.get("engine") == "none" else "starting"
    after = (
        session.get("tunnel_url"),
        session.get("status"),
        session.get("engine_ready"),
        session.get("error"),
        session.get("engine_error"),
        session.get("ui_url"),
    )
    proxy_up = _proxy_alive(int(session.get("proxy_pid") or 0))
    if after != before:
        store.upsert_session(session)
    if session.get("tunnel_url") and session.get("status") in {"starting", "ready"} and not proxy_up:
        try:
            session = ensure_proxy(store, session)
        except (OSError, RuntimeError) as exc:
            session["error"] = str(exc)
            store.upsert_session(session)
    return session


def refresh_session(store: Store, session: dict[str, Any]) -> dict[str, Any]:
    if session.get("status") not in {"starting", "ready"} or not session.get("kernel"):
        return session
    try:
        account = store.get_account(str(session["account"]))
        token = store.read_token(account)
        logs = kaggle_cli.kernel_logs(token, str(session["kernel"]))
    except (KeyError, OSError, kaggle_cli.KaggleError):
        return session
    return adopt_logs(store, session, logs)


def describe(session: dict[str, Any]) -> dict[str, Any]:
    shown = public_session(session)
    port = session.get("local_port")
    if session.get("engine") == "unsloth":
        shown["base_url"] = ""
    elif port and _proxy_alive(int(session.get("proxy_pid") or 0)):
        shown["base_url"] = f"http://127.0.0.1:{port}/v1"
    else:
        shown["base_url"] = ""
    return shown
