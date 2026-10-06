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
import tty
import urllib.parse

from kgate.remote_agent import encode_ws_frame, read_ws_frame


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


def attach(base_url: str, token: str) -> int:
    if not sys.stdin.isatty():
        raise TermError("kgate attach needs a terminal. Run it from your shell, not a pipe.")
    sock = open_terminal(base_url, token)
    size = shutil.get_terminal_size((80, 24))
    send_resize(sock, size.columns, size.lines)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    print("connected. Ctrl-] detaches. The Kaggle session keeps running.", file=sys.stderr)
    tty.setraw(fd)
    try:
        while True:
            readable, _, _ = select.select([sock, fd], [], [], 0.5)
            if fd in readable:
                data = os.read(fd, 4096)
                if not data:
                    break
                if b"\x1d" in data:
                    break
                sock.sendall(encode_ws_frame(data, opcode=0x2, mask=True))
            if sock in readable:
                opcode, payload = read_ws_frame(sock)
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    sock.sendall(encode_ws_frame(payload, opcode=0xA, mask=True))
                    continue
                if opcode in {0x1, 0x2} and payload:
                    os.write(sys.stdout.fileno(), payload)
            current = shutil.get_terminal_size((size.columns, size.lines))
            if (current.columns, current.lines) != (size.columns, size.lines):
                size = current
                send_resize(sock, size.columns, size.lines)
    except (EOFError, OSError) as exc:
        raise TermError(str(exc)) from exc
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        try:
            sock.sendall(encode_ws_frame(b"", opcode=0x8, mask=True))
        except OSError:
            pass
        sock.close()
        print("\ndetached.", file=sys.stderr)
    return 0


