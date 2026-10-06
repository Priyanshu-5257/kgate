"""Local usage dashboard. It binds to 127.0.0.1 and never leaves the laptop."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from kgate import kaggle_cli
from kgate.control import ControlError, exec_command, health
from kgate.launch import adopt_logs, describe, stop_remote
from kgate.proxy import pick_port, port_is_free
from kgate.store import Store


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KGate</title>
<style>
  :root {
    --ink: #1b1914;
    --muted: #5e584e;
    --paper: #ebe4d4;
    --panel: #f7f1e4;
    --line: #1b1914;
    --oxide: #b8431f;
    --pine: #1f6a45;
    --amber: #c47b12;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 15px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  }
  header, main { max-width: 1080px; margin: 0 auto; padding: 28px 22px; }
  header { padding-bottom: 0; display: flex; justify-content: space-between; gap: 16px; align-items: end; }
  h1 { font-size: 42px; line-height: 0.9; margin: 0; letter-spacing: -0.04em; }
  h2 { font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 12px; }
  p { margin: 6px 0; }
  .lede { color: var(--muted); max-width: 46ch; }
  button, input {
    font: 14px/1.2 ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
    color: var(--ink);
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 8px 10px;
  }
  button { cursor: pointer; background: var(--ink); color: var(--paper); }
  button.quiet { background: var(--panel); color: var(--ink); }
  button:disabled { opacity: 0.5; cursor: wait; }
  .grid { display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 18px; }
  section { border: 1px solid var(--line); background: var(--panel); padding: 16px; margin-top: 18px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th { text-align: left; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  td, th { padding: 7px 6px 7px 0; vertical-align: top; border-bottom: 1px solid #ddd4c4; }
  .bar { height: 8px; background: #ddd4c4; margin-top: 6px; }
  .bar > span { display: block; height: 100%; background: var(--pine); }
  .bar.low > span { background: var(--oxide); }
  .num { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; }
  .url { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 18px; }
  .tag { display: inline-block; border: 1px solid var(--line); padding: 1px 6px; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
  .tag.ready { background: #d7eadc; }
  .tag.starting { background: #f3e2c2; }
  .tag.stopped, .tag.error { background: #f0d2c8; }
  pre {
    white-space: pre-wrap;
    background: #1b1914;
    color: #f3ecdc;
    padding: 12px;
    min-height: 88px;
    font: 12px/1.45 ui-monospace, Menlo, Consolas, monospace;
  }
  #logs { max-height: 440px; overflow: auto; margin-top: 8px; }
  .row { display: flex; gap: 8px; }
  .row input { flex: 1; }
  .err { color: var(--oxide); }
  @media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
    h1 { font-size: 34px; }
    header { flex-direction: column; align-items: start; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>KGate</h1>
    <p class="lede">Kaggle GPU sessions, used from this laptop. Quota is per account and burns while a notebook is up.</p>
  </div>
  <button id="refresh" class="quiet" type="button">Refresh</button>
</header>
<main>
  <div class="grid">
    <section>
      <h2>Accounts</h2>
      <div id="accounts"></div>
    </section>
    <section>
      <h2>This laptop</h2>
      <div id="local"></div>
    </section>
  </div>
  <section>
    <h2>Sessions</h2>
    <div id="sessions"></div>
  </section>
  <section>
    <h2>Notebook log</h2>
    <p id="log-meta" class="lede">Connecting to the live log. Kaggle's saved log stays empty while the notebook is running.</p>
    <pre id="logs">Waiting for the live log…</pre>
  </section>
  <section>
    <h2>Run a command on the session</h2>
    <div class="row">
      <input id="cmd" value="nvidia-smi" spellcheck="false">
      <button id="run" type="button">Run</button>
    </div>
    <p id="cmd-note" class="lede"></p>
    <pre id="out">Output lands here.</pre>
  </section>
</main>
<script>
const accountsEl = document.querySelector("#accounts");
const sessionsEl = document.querySelector("#sessions");
const localEl = document.querySelector("#local");
const outEl = document.querySelector("#out");
const noteEl = document.querySelector("#cmd-note");
let current = null;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function hours(value) {
  const match = String(value == null ? "" : value).match(/[0-9]+(?:\\.[0-9]+)?/);
  return match ? Number(match[0]) : 0;
}
function accountTable(rows) {
  accountsEl.replaceChildren();
  if (!rows.length) {
    accountsEl.append(el("p", "No accounts yet. From a shell: kgate account add NAME --token-file ~/.kaggle/access_token"));
    return;
  }
  const table = el("table");
  const head = el("tr");
  ["Account", "Kaggle user", "GPU left", "Resets"].forEach((label) => head.append(el("th", label)));
  table.append(head);
  rows.forEach((row) => {
    const tr = el("tr");
    const name = el("td");
    name.append(el("div", row.name, "num"));
    if (row.is_default) name.append(el("div", "default", "lede"));
    tr.append(name);
    tr.append(el("td", row.username || "—", "num"));
    const quota = el("td");
    if (row.error) {
      quota.append(el("div", row.error, "err"));
    } else if (row.gpu) {
      quota.append(el("div", row.gpu.remaining + " of " + row.gpu.total, "num"));
      const total = hours(row.gpu.total) || 1;
      const left = hours(row.gpu.remaining);
      const pct = Math.max(0, Math.min(100, (left / total) * 100));
      const bar = el("div", "", "bar" + (pct < 20 ? " low" : ""));
      const fill = el("span");
      fill.style.width = pct.toFixed(1) + "%";
      bar.append(fill);
      quota.append(bar);
    } else {
      quota.append(el("div", "—"));
    }
    tr.append(quota);
    tr.append(el("td", (row.gpu && row.gpu.refreshAt) || "—", "num"));
    table.append(tr);
  });
  accountsEl.append(table);
}
function sessionTable(rows) {
  sessionsEl.replaceChildren();
  const live = rows.filter((row) => row.status === "ready" || row.status === "starting");
  current = live.length ? live[live.length - 1] : null;
  noteEl.textContent = current
    ? "Commands go to " + current.id + " on " + current.account + ". A full shell is `kgate attach`."
    : "No running session. Start one with `kgate up`.";
  if (!rows.length) {
    sessionsEl.append(el("p", "No sessions recorded on this machine."));
    return;
  }
  const table = el("table");
  const head = el("tr");
  ["Session", "Engine", "Status", "Local API", ""].forEach((label) => head.append(el("th", label)));
  table.append(head);
  rows.slice().reverse().forEach((row) => {
    const tr = el("tr");
    const who = el("td");
    who.append(el("div", row.id, "num"));
    who.append(el("div", (row.username || row.account || "") + " · " + (row.accelerator || ""), "lede"));
    tr.append(who);
    const engine = el("td");
    engine.append(el("div", row.engine || "none"));
    engine.append(el("div", row.model || "shell only", "num"));
    tr.append(engine);
    const status = el("td");
    status.append(el("span", row.status || "unknown", "tag " + (row.status || "")));
    if (row.engine_error) status.append(el("div", row.engine_error, "err"));
    tr.append(status);
    tr.append(el("td", row.base_url || "—", "num"));
    const actions = el("td");
    if (row.status === "ready" || row.status === "starting") {
      const stop = el("button", "Stop");
      stop.type = "button";
      stop.addEventListener("click", () => stopSession(row.id, stop));
      actions.append(stop);
    }
    tr.append(actions);
    table.append(tr);
  });
  sessionsEl.append(table);
}
function localCard(payload) {
  localEl.replaceChildren();
  const live = (payload.sessions || []).filter((row) => row.base_url && (row.status === "ready" || row.status === "starting"));
  if (!live.length) {
    localEl.append(el("p", "Nothing is forwarded to this laptop right now."));
    return;
  }
  const row = live[live.length - 1];
  localEl.append(el("div", row.base_url, "url"));
  localEl.append(el("p", "OpenAI base URL. Use any API key. Model: " + (row.model || "(none until you start one in the shell)")));
  if (row.kernel_url) {
    const link = el("a", row.kernel_url);
    link.href = row.kernel_url;
    localEl.append(link);
  }
}
async function load(refresh) {
  const response = await fetch(refresh ? "/api/overview?refresh=1" : "/api/overview");
  const payload = await response.json();
  accountTable(payload.accounts || []);
  sessionTable(payload.sessions || []);
  localCard(payload);
}
async function stopSession(id, button) {
  button.disabled = true;
  outEl.textContent = "Stopping " + id + "…";
  const response = await fetch("/api/sessions/" + encodeURIComponent(id) + "/stop", {method: "POST"});
  const payload = await response.json();
  outEl.textContent = payload.note || payload.error || JSON.stringify(payload, null, 2);
  await load(true);
}
async function runCommand() {
  if (!current) {
    outEl.textContent = "No running session.";
    return;
  }
  const command = document.querySelector("#cmd").value;
  outEl.textContent = "Running…";
  const response = await fetch("/api/sessions/" + encodeURIComponent(current.id) + "/exec", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({cmd: command})
  });
  const payload = await response.json();
  if (payload.error) {
    outEl.textContent = payload.error;
    return;
  }
  outEl.textContent = (payload.stdout || "") + (payload.stderr || "");
}
document.querySelector("#refresh").addEventListener("click", () => load(true));
document.querySelector("#run").addEventListener("click", runCommand);
document.querySelector("#cmd").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runCommand();
});
let seenBase = "";
async function loadLogs() {
  const pre = document.querySelector("#logs");
  const meta = document.querySelector("#log-meta");
  const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 32;
  let payload;
  try {
    const response = await fetch("/api/logs");
    payload = await response.json();
  } catch (err) {
    meta.textContent = "Could not refresh the log.";
    return;
  }
  meta.textContent = payload.meta || "No notebook log yet.";
  pre.textContent = payload.text || payload.error || "(no output yet)";
  if (stick) pre.scrollTop = pre.scrollHeight;
  if ((payload.base_url || "") !== seenBase) {
    seenBase = payload.base_url || "";
    load(false);
  }
}
load(false);
loadLogs();
setInterval(() => load(false), 15000);
setInterval(loadLogs, 5000);
</script>
</body>
</html>
"""


def _gpu_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("resource", "")).upper() == "GPU":
            return row
    return None


class Dashboard:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._quota_at = 0.0
        self._quota: dict[str, dict[str, Any]] = {}
        self._log_lock = threading.Lock()
        self._log_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def quotas(self, force: bool = False) -> dict[str, dict[str, Any]]:
        if not force and time.time() - self._quota_at < 45 and self._quota:
            return self._quota
        found: dict[str, dict[str, Any]] = {}
        for account in self.store.list_accounts():
            try:
                token = self.store.read_token(account)
                rows = kaggle_cli.quota(token)
                found[account["name"]] = {"gpu": _gpu_row(rows), "rows": rows, "error": ""}
            except (kaggle_cli.KaggleError, OSError, KeyError) as exc:
                found[account["name"]] = {"gpu": None, "rows": [], "error": str(exc)}
        self._quota = found
        self._quota_at = time.time()
        return found

    def overview(self, force_quota: bool = False) -> dict[str, Any]:
        default = ""
        try:
            default = self.store.default_account_name()
        except KeyError:
            default = ""
        quota = self.quotas(force_quota)
        accounts = []
        for account in self.store.list_accounts():
            item = quota.get(account["name"], {})
            accounts.append(
                {
                    "name": account["name"],
                    "username": account.get("username") or "",
                    "is_default": account["name"] == default,
                    "gpu": item.get("gpu"),
                    "error": item.get("error") or "",
                }
            )
        sessions = []
        for session in self.store.list_sessions():
            shown = describe(session)
            if session.get("status") in {"starting", "ready"} and session.get("tunnel_url"):
                try:
                    live = health(str(session["tunnel_url"]), str(session["token"]), timeout=4)
                    shown["engine_ready"] = bool(live.get("engine_ready"))
                    shown["engine_error"] = live.get("engine_error") or ""
                    shown["uptime_s"] = live.get("uptime_s")
                    shown["gpu"] = live.get("gpu") or ""
                    if live.get("engine_ready"):
                        shown["status"] = "ready"
                except ControlError as exc:
                    shown["engine_error"] = str(exc)
            sessions.append(shown)
        return {"accounts": accounts, "sessions": sessions}

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        account = self.store.get_account(str(session["account"]))
        token = self.store.read_token(account)
        note = stop_remote(self.store, session, token)
        return {"ok": True, "note": note}

    def exec(self, session_id: str, command: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session.get("status") not in {"starting", "ready"}:
            raise ControlError("that session is not running")
        if not session.get("tunnel_url"):
            raise ControlError("that session has no tunnel yet")
        return exec_command(str(session["tunnel_url"]), str(session["token"]), command, timeout=30)

    def logs(self, session_id: str | None = None) -> dict[str, Any]:
        chosen = self._log_session(session_id)
        if chosen is None:
            return {
                "session_id": "",
                "text": "",
                "meta": "No session yet. Start one with kgate up. Quota for each account is above.",
                "status": "",
                "base_url": "",
            }
        cached = self._log_cache.get(chosen["id"])
        if cached and time.time() - cached[0] < 3:
            return cached[1]
        with self._log_lock:
            cached = self._log_cache.get(chosen["id"])
            if cached and time.time() - cached[0] < 3:
                return cached[1]
            try:
                account = self.store.get_account(str(chosen["account"]))
                text = kaggle_cli.kernel_logs(
                    self.store.read_token(account),
                    str(chosen["kernel"]),
                    idle_s=1.0,
                    max_s=12,
                )
                chosen = adopt_logs(self.store, chosen, text)
            except (KeyError, OSError, kaggle_cli.KaggleError) as exc:
                payload = {
                    "session_id": chosen["id"],
                    "text": "",
                    "error": str(exc),
                    "meta": f"{chosen.get('kernel', '')} · log stream failed",
                    "status": chosen.get("status") or "",
                    "base_url": "",
                }
                self._log_cache[chosen["id"]] = (time.time(), payload)
                return payload
        shown = describe(chosen)
        lines = text.splitlines()
        tail = "\n".join(lines[-400:])
        payload = {
            "session_id": shown["id"],
            "kernel": shown.get("kernel") or "",
            "text": tail,
            "status": shown.get("status") or "",
            "base_url": shown.get("base_url") or "",
            "meta": (
                f"{shown.get('kernel', '')} · {shown.get('status', '')}"
                + (f" · {shown['base_url']}" if shown.get("base_url") else "")
                + " · live stream"
            ),
        }
        self._log_cache[str(shown["id"])] = (time.time(), payload)
        return payload

    def _log_session(self, session_id: str | None) -> dict[str, Any] | None:
        if session_id:
            try:
                return self.store.get_session(session_id)
            except KeyError:
                return None
        rows = self.store.list_sessions()
        live = [row for row in rows if row.get("status") in {"starting", "ready"}]
        if live:
            return live[-1]
        return rows[-1] if rows else None


class DashHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    @property
    def dash(self) -> Dashboard:
        server = self.server
        assert isinstance(server, DashServer)
        return server.dash

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        if parts.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parts.path == "/api/overview":
            force = "refresh=1" in (parts.query or "")
            self._json(200, self.dash.overview(force))
            return
        if parts.path == "/api/logs":
            self._json(200, self.dash.logs())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        bits = [bit for bit in parts.path.split("/") if bit]
        try:
            if len(bits) == 4 and bits[0] == "api" and bits[1] == "sessions" and bits[3] == "stop":
                self._json(200, self.dash.stop(bits[2]))
                return
            if len(bits) == 4 and bits[0] == "api" and bits[1] == "sessions" and bits[3] == "exec":
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw.decode() or "{}")
                self._json(200, self.dash.exec(bits[2], str(payload.get("cmd") or "")))
                return
        except KeyError:
            self._json(404, {"error": "unknown session"})
            return
        except (ControlError, kaggle_cli.KaggleError, OSError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})


class DashServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, store: Store) -> None:
        self.dash = Dashboard(store)
        super().__init__(("127.0.0.1", port), DashHandler)


def _listener_pid(port: int) -> int | None:
    needle = f":{port:04X}"
    inodes: set[str] = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        if not os.path.exists(name):
            continue
        with open(name, encoding="utf-8", errors="replace") as handle:
            next(handle, None)
            for line in handle:
                fields = line.split()
                if len(fields) < 10 or not fields[1].upper().endswith(needle):
                    continue
                if fields[3] != "0A":  # listening
                    continue
                inodes.add(fields[9])
    if not inodes:
        return None
    proc = "/proc"
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        fd_dir = os.path.join(proc, pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            for inode in inodes:
                if target == f"socket:[{inode}]":
                    return int(pid)
    return None


def _cmdline(pid: int) -> str:
    path = f"/proc/{pid}/cmdline"
    try:
        with open(path, "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def ensure_dashboard(store: Store) -> str:
    """Start the local dashboard if it is not already this version."""
    port = int(store.config().get("dashboard_port") or 8787)
    url = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            page = response.read().decode("utf-8", errors="replace")
        if "Notebook log" in page:
            return url
        pid = _listener_pid(port)
        if pid and "kgate" in _cmdline(pid):
            os.kill(pid, 15)
            for _ in range(20):
                if not _listener_pid(port):
                    break
                time.sleep(0.1)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    if not port_is_free(port):
        port = pick_port(port)
        url = f"http://127.0.0.1:{port}"
    log_path = store.log_dir / "dashboard.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "kgate", "dash", "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(25):
        try:
            with urllib.request.urlopen(url, timeout=0.4) as response:
                if response.status == 200:
                    return url
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.2)
    return url


def serve_dashboard(store: Store, port: int) -> None:
    server = DashServer(port, store)
    print(f"Dashboard at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.", flush=True)


