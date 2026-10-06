"""Interactive shell over the session websocket."""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import socket
import ssl
import sys
import termios
import time
import tty
import urllib.parse

from kgate.remote_agent import encode_ws_frame, read_ws_frame

PING_EVERY_S = 15
RECONNECTS = 8


class TermError(RuntimeError):
    pass


def open_terminal(base_url: str, token: str, timeout: float = 20) -> socket.socket:
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise TermError(f"bad session URL {base_url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    raw = socket.create_connection((parts.hostname, port), timeout=timeout)
    if parts.scheme == "https":
        raw = ssl.create_default_context().wrap_socket(raw, server_hostname=parts.hostname)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    raw.settimeout(timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    host_header = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
    request = (
        "GET /kgate/term/ws HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Authorization: Bearer {token}\r\n"
        "\r\n"
    )
    raw.sendall(request.encode())
    head = b""
    while b"\r\n\r\n" not in head:
        piece = raw.recv(1)
        if not piece:
            raise TermError("tunnel closed during the terminal handshake")
        head += piece
        if len(head) > 8192:
            raise TermError("terminal handshake was too large")
    status = head.split(b"\r\n", 1)[0]
    if b" 101 " not in status:
        raise TermError("terminal handshake failed: " + status.decode("utf-8", errors="replace"))
    raw.settimeout(None)
    return raw


def send_resize(sock: socket.socket, cols: int, rows: int) -> None:
    payload = json.dumps({"type": "resize", "cols": cols, "rows": rows}).encode()
    sock.sendall(encode_ws_frame(payload, opcode=0x1, mask=True))


def tls_pending(sock: socket.socket) -> int:
    """Bytes OpenSSL already decrypted. select() cannot see these."""
    pending = getattr(sock, "pending", None)
    if not callable(pending):
        return 0
    try:
        return int(pending())
    except (OSError, ValueError):
        return 0


def _note(text: str) -> None:
    os.write(sys.stderr.fileno(), f"\r\n[kgate] {text}\r\n".encode())


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.sendall(encode_ws_frame(b"", opcode=0x8, mask=True))
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def pump(sock: socket.socket, fd: int) -> bool:
    """Copy the local terminal and the remote shell.

    Returns False when the user detaches with Ctrl-]. Returns True when the
    tunnel drops and the caller should open it again. The remote bash is left
    running in both cases.
    """
    size = shutil.get_terminal_size((80, 24))
    send_resize(sock, size.columns, size.lines)
    last_ping = time.monotonic()
    out = sys.stdout.fileno()
    try:
        while True:
            now = time.monotonic()
            if now - last_ping >= PING_EVERY_S:
                sock.sendall(encode_ws_frame(b"k", opcode=0x9, mask=True))
                last_ping = now
            if tls_pending(sock):
                readable = [sock]
                if select.select([fd], [], [], 0)[0]:
                    readable.append(fd)
            else:
                readable, _, _ = select.select([sock, fd], [], [], 1.0)
            if fd in readable:
                data = os.read(fd, 4096)
                if not data or b"\x1d" in data:
                    return False
                sock.sendall(encode_ws_frame(data, opcode=0x2, mask=True))
            if sock in readable:
                opcode, payload = read_ws_frame(sock)
                if opcode == 0x8:
                    return True
                if opcode == 0x9:
                    sock.sendall(encode_ws_frame(payload, opcode=0xA, mask=True))
                    continue
                if opcode in {0x1, 0x2} and payload:
                    os.write(out, payload)
            current = shutil.get_terminal_size((size.columns, size.lines))
            if (current.columns, current.lines) != (size.columns, size.lines):
                size = current
                send_resize(sock, size.columns, size.lines)
    except (EOFError, OSError, ValueError):
        return True


def attach(base_url: str, token: str) -> int:
    if not sys.stdin.isatty():
        raise TermError("kgate attach needs a terminal. Run it from your shell, not a pipe.")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    print("connected. Ctrl-] detaches. A dropped tunnel reconnects to the same shell.", file=sys.stderr)
    tty.setraw(fd)
    failures = 0
    user_left = False
    try:
        while failures < RECONNECTS:
            started = time.monotonic()
            try:
                sock = open_terminal(base_url, token, timeout=15)
            except TermError:
                failures += 1
                _note(f"tunnel unreachable ({failures}/{RECONNECTS})")
                time.sleep(min(3.0, 0.4 * failures))
                continue
            try:
                lost = pump(sock, fd)
            finally:
                _close_socket(sock)
            if not lost:
                user_left = True
                return 0
            if time.monotonic() - started > 5:
                failures = 0
            failures += 1
            _note("tunnel dropped, reconnecting to the same shell")
            time.sleep(0.4)
        return 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        print("\ndetached." if user_left else "\nconnection lost.", file=sys.stderr)


