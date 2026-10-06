"""Session agent that runs inside a Kaggle kernel.

This file is uploaded as the kernel script. It stays on the Python standard
library so the notebook does not need this repo installed. A launcher prepends
``CONFIG = json.loads(...)`` before the source. Local tests import the helpers
and the gateway without that assignment.
"""

import hashlib
import hmac
import http.client
import json
import os
import pty
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_BODY = 32 * 1024 * 1024
_CHILDREN: list[subprocess.Popen[Any]] = []


def log(message: str) -> None:
    print(f"[kgate] {message}", flush=True)


def emit(kind: str, payload: dict[str, Any]) -> None:
    print(f"KGATE_{kind} {json.dumps(payload, separators=(',', ':'))}", flush=True)


def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len([line for line in out.splitlines() if line.strip()])


def gpu_summary() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"


def cloudflared_url_from_text(text: str) -> str | None:
    import re

    match = re.search(r"https://[A-Za-z0-9-]+\.trycloudflare\.com", text)
    return match.group(0) if match else None


def engine_argv(cfg: dict[str, Any], gpus: int) -> list[str]:
    engine = cfg["engine"]
    port = str(cfg["engine_port"])
    model = cfg["model"]
    extra = list(cfg.get("engine_args") or [])
    raw_tp = cfg.get("tensor_parallel")
    tp = 1 if raw_tp is None else int(raw_tp)
    if tp <= 0:
        tp = max(1, gpus)
    if engine == "vllm":
        argv = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--dtype",
            cfg.get("dtype") or "half",
            "--max-model-len",
            str(cfg.get("max_model_len") or 4096),
            "--gpu-memory-utilization",
            str(cfg.get("gpu_memory_utilization") or 0.9),
        ]
        if cfg.get("trust_remote_code", True):
            argv.append("--trust-remote-code")
        if cfg.get("quantization"):
            argv.extend(["--quantization", str(cfg["quantization"])])
        if tp > 1:
            argv.extend(["--tensor-parallel-size", str(tp)])
        return argv + extra
    if engine == "sglang":
        dtype = cfg.get("dtype") or "float16"
        if dtype == "half":
            dtype = "float16"
        argv = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--dtype",
            dtype,
        ]
        if tp > 1:
            argv.extend(["--tp", str(tp)])
        return argv + extra
    if engine == "ollama":
        return ["ollama", "serve"]
    if engine == "none":
        return []
    raise ValueError(f"unknown engine {engine}")


def pip_packages(engine: str) -> list[str]:
    if engine == "vllm":
        return ["vllm"]
    if engine == "sglang":
        return ["sglang"]
    return []


def encode_ws_frame(data: bytes, opcode: int = 0x2, mask: bool = False) -> bytes:
    head = bytes([0x80 | (opcode & 0x0F)])
    length = len(data)
    mask_bit = 0x80 if mask else 0
    if length < 126:
        head += bytes([mask_bit | length])
    elif length < 65536:
        head += bytes([mask_bit | 126]) + struct.pack("!H", length)
    else:
        head += bytes([mask_bit | 127]) + struct.pack("!Q", length)
    if not mask:
        return head + data
    key = os.urandom(4)
    masked = bytes(byte ^ key[index % 4] for index, byte in enumerate(data))
    return head + key + masked


def read_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        piece = sock.recv(remaining)
        if not piece:
            raise EOFError("socket closed")
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def read_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = read_exact(sock, 2)
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exact(sock, 8))[0]
    if length > MAX_BODY:
        raise ValueError("websocket frame is too large")
    key = read_exact(sock, 4) if masked else b""
    data = read_exact(sock, length) if length else b""
    if masked:
        data = bytes(byte ^ key[index % 4] for index, byte in enumerate(data))
    return opcode, data


def token_ok(header_value: str | None, token: str) -> bool:
    if not header_value or not token:
        return False
    scheme, _, presented = header_value.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return hmac.compare_digest(presented.strip(), token)


class ShellSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.master: int | None = None
        self.proc: subprocess.Popen[Any] | None = None

    def ensure(self, cols: int = 120, rows: int = 32) -> int:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None and self.master is not None:
                return self.master
            self._close_locked()
            master, slave = pty.openpty()
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = str(cols)
            env["LINES"] = str(rows)
            self.proc = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-i"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=env,
                preexec_fn=os.setsid,
                close_fds=True,
            )
            os.close(slave)
            self.master = master
            self.resize(rows, cols)
            return master

    def resize(self, rows: int, cols: int) -> None:
        if self.master is None:
            return
        import fcntl
        import termios

        window = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
        fcntl.ioctl(self.master, termios.TIOCSWINSZ, window)

    def _close_locked(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=2)
        if self.master is not None:
            try:
                os.close(self.master)
            except OSError:
                pass
        self.master = None
        self.proc = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()


def _track(proc: subprocess.Popen[Any]) -> subprocess.Popen[Any]:
    _CHILDREN.append(proc)
    return proc


def stop_children() -> None:
    for proc in list(_CHILDREN):
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
    deadline = time.time() + 5
    for proc in list(_CHILDREN):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()


class GateHandler(BaseHTTPRequestHandler):
    rbufsize = 0
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    @property
    def gate(self) -> "GateServer":
        server = self.server
        assert isinstance(server, GateServer)
        return server

    def _authorized(self) -> bool:
        return token_ok(self.headers.get("Authorization"), self.gate.token)

    def _deny(self) -> None:
        body = b"unauthorized\n"
        self.send_response(401)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        if length < 0 or length > MAX_BODY:
            raise ValueError("body is too large")
        if length == 0:
            return b""
        return self.rfile.read(length)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._route("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._route("DELETE")

    def _route(self, method: str) -> None:
        if not self._authorized():
            self._deny()
            return
        path = urlsplit(self.path).path
        try:
            if path == "/kgate/health" and method == "GET":
                self._health()
            elif path == "/kgate/gpu" and method == "GET":
                self._text(gpu_summary())
            elif path == "/kgate/stop" and method == "POST":
                self._send_json(200, {"ok": True, "stopping": True})
                threading.Thread(target=self.gate.request_stop, daemon=True).start()
            elif path == "/kgate/exec" and method == "POST":
                self._exec()
            elif path == "/kgate/term/ws" and method == "GET":
                self._websocket()
            elif path.startswith("/kgate/"):
                self._send_json(404, {"ok": False, "error": "not found"})
            else:
                self._proxy(method)
        except Exception as exc:  # noqa: BLE001 — report to the caller, keep the session up
            try:
                self._send_json(500, {"ok": False, "error": str(exc)})
            except Exception:
                pass

    def _text(self, text: str) -> None:
        body = (text + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _health(self) -> None:
        state = self.gate.state
        tail = ""
        log_path = state.get("log_path") or ""
        if log_path and os.path.exists(log_path):
            with open(log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4000))
                tail = handle.read().decode("utf-8", errors="replace")
        self._send_json(
            200,
            {
                "ok": True,
                "run": state.get("run"),
                "engine": state.get("engine"),
                "model": state.get("model"),
                "engine_ready": bool(state.get("engine_ready")),
                "engine_error": state.get("engine_error") or "",
                "uptime_s": int(time.time() - float(state.get("started") or time.time())),
                "gpu": gpu_summary(),
                "engine_log_tail": tail[-2000:],
            },
        )

    def _exec(self) -> None:
        raw = self._read_body()
        payload = json.loads(raw.decode() or "{}")
        command = str(payload.get("cmd") or "")
        timeout = float(payload.get("timeout") or 30)
        if not command.strip():
            self._send_json(400, {"ok": False, "error": "cmd is required"})
            return
        timeout = min(max(timeout, 1), 120)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            code = completed.returncode
            stdout = completed.stdout[:200_000]
            stderr = completed.stderr[:200_000]
        except subprocess.TimeoutExpired as exc:
            code = 124
            stdout = exc.stdout or b""
            stderr = (exc.stderr or b"") + b"\ntimeout\n"
        self._send_json(
            200,
            {
                "ok": code == 0,
                "code": code,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            },
        )

    def _proxy(self, method: str) -> None:
        body = self._read_body() if method in {"POST", "PUT", "PATCH"} else None
        headers = {}
        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]
        if self.headers.get("Accept"):
            headers["Accept"] = self.headers["Accept"]
        conn = http.client.HTTPConnection("127.0.0.1", self.gate.engine_port, timeout=600)
        try:
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()
        except OSError as exc:
            self._send_json(502, {"ok": False, "error": f"engine is not accepting connections: {exc}"})
            return
        self.send_response(response.status)
        skip = {"transfer-encoding", "connection", "keep-alive", "content-length"}
        for key, value in response.headers.items():
            if key.lower() not in skip:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = response.read(16384)
            if not chunk:
                break
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        conn.close()

    def _websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._send_json(400, {"ok": False, "error": "missing websocket key"})
            return
        accept = hashlib.sha1((key + WS_GUID).encode()).digest()
        import base64

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", base64.b64encode(accept).decode())
        self.end_headers()
        self.wfile.flush()
        sock = self.connection
        sock.settimeout(None)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        shell = self.gate.shell
        master = shell.ensure()
        try:
            while True:
                readable, _, _ = select.select([sock, master], [], [], 0.5)
                if sock in readable:
                    opcode, data = read_ws_frame(sock)
                    if opcode == 0x8:
                        break
                    if opcode == 0x9:
                        sock.sendall(encode_ws_frame(data, opcode=0xA, mask=False))
                        continue
                    if opcode == 0x1:
                        self._control_text(data, shell)
                        continue
                    if opcode == 0x2 and data:
                        os.write(master, data)
                if master in readable:
                    try:
                        data = os.read(master, 8192)
                    except OSError:
                        break
                    if not data:
                        break
                    sock.sendall(encode_ws_frame(data, opcode=0x2, mask=False))
        except (EOFError, OSError, ValueError):
            pass
        finally:
            self.close_connection = True

    def _control_text(self, data: bytes, shell: ShellSession) -> None:
        try:
            payload = json.loads(data.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if payload.get("type") == "resize":
            shell.resize(int(payload.get("rows") or 24), int(payload.get("cols") or 80))


class GateServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], token: str, engine_port: int, state: dict[str, Any]) -> None:
        self.token = token
        self.engine_port = engine_port
        self.state = state
        self.shell = ShellSession()
        self._stop = threading.Event()
        super().__init__(address, GateHandler)

    def request_stop(self) -> None:
        self._stop.set()
        self.shell.close()
        stop_children()
        time.sleep(0.2)
        os._exit(0)


def install_cloudflared() -> str:
    destination = "/tmp/kgate-cloudflared"
    if os.path.isfile(destination) and os.access(destination, os.X_OK):
        return destination
    machine = os.uname().machine
    arch = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    urls = [
        f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}",
    ]
    last_error = "download failed"
    for url in urls:
        try:
            urllib.request.urlretrieve(url, destination)  # noqa: S310 — fixed vendor URL
            os.chmod(destination, 0o755)
            return destination
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    raise RuntimeError(f"could not download cloudflared: {last_error}")


def start_tunnel(local_port: int) -> tuple[subprocess.Popen[Any], str]:
    binary = install_cloudflared()
    proc = _track(
        subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{local_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    )
    assert proc.stdout is not None
    deadline = time.time() + 90
    collected: list[str] = []
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.2)
            continue
        collected.append(line)
        print(line, end="", flush=True)
        found = cloudflared_url_from_text(line) or cloudflared_url_from_text("".join(collected))
        if found:
            return proc, found
    tail = "".join(collected)[-2000:]
    raise RuntimeError(f"cloudflared did not return a tunnel URL.\n{tail}")


def pip_install(packages: list[str]) -> None:
    if not packages:
        return
    log("installing " + " ".join(packages))
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *packages],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pip install failed with exit {completed.returncode}")


def install_ollama() -> None:
    if shutil.which("ollama"):
        return
    log("installing ollama")
    completed = subprocess.run(
        ["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"],
        check=False,
    )
    if completed.returncode != 0 or not shutil.which("ollama"):
        raise RuntimeError("ollama install failed")


def apply_hf_token(cfg: dict[str, Any]) -> None:
    token = cfg.get("hf_token") or ""
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def wait_http(port: int, path: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            if response.status < 500:
                return True
        except OSError:
            pass
        finally:
            conn.close()
        time.sleep(2)
    return False


def start_engine(cfg: dict[str, Any], state: dict[str, Any]) -> subprocess.Popen[Any] | None:
    engine = cfg["engine"]
    state["engine"] = engine
    state["model"] = cfg.get("model") or ""
    if engine == "none":
        state["engine_ready"] = True
        emit("ENGINE", {"run": cfg["run_id"], "ready": True, "engine": "none"})
        return None
    apply_hf_token(cfg)
    log_path = state["log_path"]
    log_handle = open(log_path, "ab", buffering=0)
    gpus = gpu_count()
    log(f"visible GPUs: {gpus}")
    env = os.environ.copy()
    if engine == "ollama":
        install_ollama()
        env["OLLAMA_HOST"] = f"127.0.0.1:{cfg['engine_port']}"
        proc = _track(
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        )
        if not wait_http(int(cfg["engine_port"]), "/api/tags", 120):
            raise RuntimeError("ollama serve did not open its port")
        log(f"pulling {cfg['model']}")
        pull = subprocess.run(
            ["ollama", "pull", cfg["model"]],
            env=env,
            check=False,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"ollama pull failed with exit {pull.returncode}")
        state["engine_ready"] = True
        emit("ENGINE", {"run": cfg["run_id"], "ready": True, "engine": "ollama", "model": cfg["model"]})
        return proc

    pip_install(pip_packages(engine))
    argv = engine_argv(cfg, gpus)
    log("starting " + " ".join(argv))
    proc = _track(
        subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    )

    def _watch() -> None:
        ready = wait_http(int(cfg["engine_port"]), "/v1/models", float(cfg.get("engine_timeout_s") or 2400))
        if proc.poll() is not None:
            state["engine_ready"] = False
            state["engine_error"] = f"engine exited {proc.returncode}"
            emit("ENGINE", {"run": cfg["run_id"], "ready": False, "error": state["engine_error"]})
            return
        state["engine_ready"] = ready
        if ready:
            emit("ENGINE", {"run": cfg["run_id"], "ready": True, "engine": engine, "model": cfg["model"]})
        else:
            state["engine_error"] = "engine did not become ready before the timeout"
            emit("ENGINE", {"run": cfg["run_id"], "ready": False, "error": state["engine_error"]})

    threading.Thread(target=_watch, daemon=True).start()
    return proc


def serve_gateway(cfg: dict[str, Any], state: dict[str, Any]) -> GateServer:
    server = GateServer(("127.0.0.1", int(cfg["gate_port"])), cfg["token"], int(cfg["engine_port"]), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def load_config() -> dict[str, Any]:
    cfg = globals().get("CONFIG")
    if isinstance(cfg, dict):
        return cfg
    raw = os.environ.get("KGATE_CONFIG")
    if raw:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    raise SystemExit("kgate agent is missing CONFIG")


def main() -> None:
    cfg = load_config()
    run_id = str(cfg["run_id"])
    state: dict[str, Any] = {
        "run": run_id,
        "engine": cfg.get("engine") or "none",
        "model": cfg.get("model") or "",
        "engine_ready": False,
        "engine_error": "",
        "started": time.time(),
        "log_path": "/kaggle/working/kgate-engine.log"
        if os.path.isdir("/kaggle/working")
        else os.path.join(tempfile_dir(), "kgate-engine.log"),
    }
    emit("BOOT", {"run": run_id, "engine": state["engine"], "model": state["model"]})
    log(gpu_summary())
    server = serve_gateway(cfg, state)
    try:
        _proc, url = start_tunnel(int(cfg["gate_port"]))
    except Exception as exc:  # noqa: BLE001
        emit("TUNNEL", {"run": run_id, "error": str(exc)})
        log(str(exc))
        raise
    emit("TUNNEL", {"run": run_id, "url": url})
    log(f"tunnel {url}")
    try:
        start_engine(cfg, state)
    except Exception as exc:  # noqa: BLE001
        state["engine_error"] = str(exc)
        state["engine_ready"] = False
        emit("ENGINE", {"run": cfg["run_id"], "ready": False, "error": str(exc)})
        log(str(exc))
    started = time.time()
    die_at = started + float(cfg.get("die_after_s") or 6 * 3600)
    ticks = 0
    while time.time() < die_at:
        time.sleep(15)
        ticks += 1
        if ticks % 4 == 0:
            ready = "yes" if state.get("engine_ready") else "no"
            log(f"alive engine_ready={ready} uptime_s={int(time.time() - started)}")
    log("session time limit reached, stopping")
    server.request_stop()


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


if __name__ == "__main__":
    main()
