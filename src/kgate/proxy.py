"""Local OpenAI-compatible reverse proxy.

Binds to 127.0.0.1 only. Adds the session bearer token and streams the
response, so chat completions work from curl or any OpenAI client pointed
at http://127.0.0.1:<port>/v1.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

MAX_BODY = 32 * 1024 * 1024
HOP = {"transfer-encoding", "connection", "keep-alive", "content-length", "proxy-connection"}


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, base_url: str, token: str) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"bad tunnel URL: {base_url}")
        self.base_scheme = parts.scheme
        self.base_host = parts.hostname
        self.base_port = parts.port or (443 if parts.scheme == "https" else 80)
        self.token = token
        super().__init__(("127.0.0.1", port), ProxyHandler)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[proxy]", (fmt % args) if args else fmt, flush=True)

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_PUT(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")

    def _forward(self) -> None:
        server = self.server
        assert isinstance(server, ProxyServer)
        length = int(self.headers.get("Content-Length") or "0")
        if length < 0 or length > MAX_BODY:
            self.send_error(413, "body too large")
            return
        body = self.rfile.read(length) if length else None
        headers = {"Authorization": f"Bearer {server.token}", "Host": server.base_host}
        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]
        if self.headers.get("Accept"):
            headers["Accept"] = self.headers["Accept"]
        conn: http.client.HTTPConnection
        if server.base_scheme == "https":
            context = ssl.create_default_context()
            conn = http.client.HTTPSConnection(server.base_host, server.base_port, timeout=600, context=context)
        else:
            conn = http.client.HTTPConnection(server.base_host, server.base_port, timeout=600)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
        except OSError as exc:
            message = json.dumps({"error": f"session tunnel is unreachable: {exc}"}).encode()
            self.send_response(502)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return
        self.send_response(response.status)
        self._cors()
        for key, value in response.headers.items():
            if key.lower() not in HOP:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = response.read(16384)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()


def load_session_endpoint(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    url = payload.get("tunnel_url") or ""
    token = payload.get("token") or ""
    if not url or not token:
        raise SystemExit(f"{path} has no tunnel_url or token")
    return str(url), str(token)


def serve(port: int, base_url: str, token: str) -> None:
    server = ProxyServer(port, base_url, token)
    print(f"[proxy] 127.0.0.1:{port} -> {base_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local proxy in front of a kgate session")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--session-file", required=True)
    args = parser.parse_args(argv)
    base_url, token = load_session_endpoint(args.session_file)
    serve(args.port, base_url, token)
    return 0


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        if port_is_free(port):
            return port
    raise RuntimeError(f"no free local port near {preferred}")


def spawn(port: int, session_file: str, log_file: str) -> int:
    import subprocess
    import sys

    log = open(log_file, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "kgate.proxy", "--port", str(port), "--session-file", session_file],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    return proc.pid


def stop_pid(pid: int) -> None:
    import signal

    if pid <= 0:
        return
    cmdline = ""
    proc_path = f"/proc/{pid}/cmdline"
    if os.path.exists(proc_path):
        with open(proc_path, "rb") as handle:
            cmdline = handle.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    if "kgate.proxy" not in cmdline and "kgate/proxy" not in cmdline:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
