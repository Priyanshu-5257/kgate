"""Local tests. They do not start a Kaggle kernel or spend GPU quota."""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kgate.launch import kernel_metadata, render_session_source
from kgate.proxy import ProxyServer
from kgate.remote_agent import (
    GateServer,
    cloudflared_url_from_text,
    encode_ws_frame,
    engine_argv,
    read_ws_frame,
    token_ok,
)
from kgate.store import Store
from kgate.term import open_terminal
from kgate.util import parse_hours, parse_kernel_status, parse_marker, parse_push_result, plan_hours, public_session


class ParseTests(unittest.TestCase):
    def test_status_and_push(self) -> None:
        status, failure = parse_kernel_status('user/kgate-session has status "RUNNING"\n')
        self.assertEqual(status, "RUNNING")
        self.assertEqual(failure, "")
        enum_status, _ = parse_kernel_status(
            'aivenger1st/kgate-session has status "KernelWorkerStatus.RUNNING"\n'
        )
        self.assertEqual(enum_status, "RUNNING")
        pushed = parse_push_result(
            "Kernel version 4 successfully pushed.  Please check progress at "
            "https://www.kaggle.com/code/user/kgate-session"
        )
        self.assertTrue(pushed["ok"])
        self.assertEqual(pushed["version"], 4)
        failed = parse_push_result("Kernel push error: quota exceeded")
        self.assertFalse(failed["ok"])
        self.assertIn("quota", failed["error"])

    def test_markers_ignore_other_runs(self) -> None:
        logs = "\n".join(
            [
                'noise KGATE_BOOT {"run":"old","engine":"none","model":""}',
                'KGATE_TUNNEL {"run":"old","url":"https://old-name.trycloudflare.com"}',
                'KGATE_BOOT {"run":"abc","engine":"vllm","model":"demo"}',
                'KGATE_TUNNEL {"run":"abc","url":"https://new-name.trycloudflare.com"}',
            ]
        )
        tunnel = parse_marker(logs, "TUNNEL", "abc")
        self.assertIsNotNone(tunnel)
        assert tunnel is not None
        self.assertEqual(tunnel["url"], "https://new-name.trycloudflare.com")
        self.assertIsNone(parse_marker(logs, "TUNNEL", "missing"))

    def test_cloudflared_line(self) -> None:
        sample = "INF |  https://random-words.trycloudflare.com  |"
        self.assertEqual(cloudflared_url_from_text(sample), "https://random-words.trycloudflare.com")

    def test_plan_hours_caps_to_quota(self) -> None:
        hours, warning = plan_hours(6, 29.5)
        self.assertEqual(hours, 6)
        self.assertEqual(warning, "")
        hours, warning = plan_hours(6, 1.2)
        self.assertLess(hours, 1.2)
        self.assertIn("1.20h", warning)
        with self.assertRaises(ValueError):
            plan_hours(6, 0)

    def test_parse_hours(self) -> None:
        self.assertEqual(parse_hours("29.58h"), 29.58)


class BuildTests(unittest.TestCase):
    def test_session_script_embeds_config_as_data(self) -> None:
        cfg = {
            "run_id": "abc",
            "token": 'sek"ret',
            "model": "org/model\nname",
            "engine": "vllm",
        }
        source = render_session_source(cfg)
        compile(source, "session.py", "exec")
        namespace = {"__name__": "kgate_session_under_test"}
        exec(source, namespace)  # noqa: S102 — the generated kernel is the object under test
        self.assertEqual(namespace["CONFIG"]["model"], "org/model\nname")
        self.assertEqual(namespace["CONFIG"]["token"], 'sek"ret')
        self.assertNotIn("print(CONFIG", source)

    def test_metadata_is_private_and_online(self) -> None:
        meta = kernel_metadata("someone", "kgate-session", "NvidiaTeslaT4")
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_internet"])
        self.assertTrue(meta["enable_gpu"])
        self.assertGreaterEqual(len(meta["title"]), 5)
        self.assertEqual(meta["id"], "someone/kgate-session")

    def test_vllm_command_uses_half_on_small_gpus(self) -> None:
        argv = engine_argv(
            {
                "engine": "vllm",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "engine_port": 8000,
                "dtype": "half",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.9,
                "trust_remote_code": True,
                "quantization": "",
                "engine_args": ["--enforce-eager"],
                "tensor_parallel": 0,
            },
            2,
        )
        self.assertIn("--dtype", argv)
        self.assertIn("half", argv)
        self.assertIn("--tensor-parallel-size", argv)
        self.assertIn("2", argv)
        self.assertIn("--enforce-eager", argv)

    def test_token_compare(self) -> None:
        self.assertTrue(token_ok("Bearer secret", "secret"))
        self.assertFalse(token_ok("Bearer secret", "other"))
        self.assertFalse(token_ok("secret", "secret"))


class StoreTests(unittest.TestCase):
    def test_roundtrip_hides_token_from_account_list(self) -> None:
        home = Path(self.id().replace(".", "_"))
        # The test home must not live in the repo. Use a temp directory.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            path = store.write_token("main", "KGAT_test_token")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            store.put_account(
                {"name": "main", "username": "user", "token_file": str(path), "created_at": "t"}
            )
            raw = (store.home / "accounts.json").read_text(encoding="utf-8")
            self.assertNotIn("KGAT_test_token", raw)
            self.assertEqual(store.read_token(store.get_account("main")), "KGAT_test_token")
            session = {
                "id": "s1",
                "account": "main",
                "status": "ready",
                "token": "KGAT_test_token",
                "started_at": "t",
            }
            store.upsert_session(session)
            shown = public_session(store.get_session("s1"))
            self.assertNotIn("token", shown)
        del home


class _EngineHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        body = b'{"data":[{"id":"demo"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ThreadingHTTPServer(("127.0.0.1", 0), _EngineHandler)
        self.engine_thread = threading.Thread(target=self.engine.serve_forever, daemon=True)
        self.engine_thread.start()
        self.token = "test-token"
        self.gate = GateServer(
            ("127.0.0.1", 0),
            self.token,
            self.engine.server_address[1],
            {"run": "abc", "engine": "vllm", "model": "demo", "engine_ready": True, "engine_error": "", "started": time.time(), "log_path": ""},
        )
        self.gate_thread = threading.Thread(target=self.gate.serve_forever, daemon=True)
        self.gate_thread.start()
        gate_port = self.gate.server_address[1]
        self.proxy = ProxyServer(0, f"http://127.0.0.1:{gate_port}", self.token)
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()

    def tearDown(self) -> None:
        self.proxy.shutdown()
        self.gate.shutdown()
        self.engine.shutdown()
        self.proxy.server_close()
        self.gate.server_close()
        self.engine.server_close()
        self.gate.shell.close()

    def test_proxy_hides_the_session_and_requires_a_token_upstream(self) -> None:
        self.assertEqual(self.proxy.server_address[0], "127.0.0.1")
        proxy_port = self.proxy.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/v1/models", timeout=5) as response:
            body = json.loads(response.read().decode())
        self.assertEqual(body["data"][0]["id"], "demo")
        gate_port = self.gate.server_address[1]
        request = urllib.request.Request(f"http://127.0.0.1:{gate_port}/v1/models")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 401)

    def test_exec_and_shell(self) -> None:
        gate_port = self.gate.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{gate_port}/kgate/exec",
            data=json.dumps({"cmd": "echo kgate-exec-ok", "timeout": 10}).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode())
        self.assertEqual(payload["code"], 0)
        self.assertIn("kgate-exec-ok", payload["stdout"])

        sock = open_terminal(f"http://127.0.0.1:{gate_port}", self.token, timeout=5)
        try:
            sock.settimeout(0.5)
            sock.sendall(encode_ws_frame(b"echo kgate-ws-ok\n", opcode=0x2, mask=True))
            collected = b""
            deadline = time.time() + 8
            while time.time() < deadline and b"kgate-ws-ok" not in collected:
                try:
                    opcode, data = read_ws_frame(sock)
                except (TimeoutError, socket.timeout):
                    continue
                if opcode in {0x1, 0x2}:
                    collected += data
            self.assertIn(b"kgate-ws-ok", collected)
        finally:
            sock.close()


class DashboardTests(unittest.TestCase):
    def test_overview_does_not_leak_the_session_token(self) -> None:
        import tempfile

        from kgate.dashboard import DashServer

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            store.upsert_session(
                {
                    "id": "s1",
                    "account": "main",
                    "username": "user",
                    "status": "ready",
                    "token": "super-secret-token",
                    "tunnel_url": "",
                    "local_port": 8000,
                    "engine": "none",
                    "model": "",
                    "kernel_url": "https://www.kaggle.com/code/user/kgate-session",
                    "accelerator": "NvidiaTeslaT4",
                    "started_at": "t",
                }
            )
            server = DashServer(0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    page = response.read().decode()
                self.assertIn("KGate", page)
                self.assertIn("Refresh", page)
                self.assertNotIn("super-secret-token", page)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/overview", timeout=5) as response:
                    payload = json.loads(response.read().decode())
                self.assertEqual(payload["sessions"][0]["base_url"], "")
                self.assertIn("Notebook log", page)
                self.assertNotIn("token", payload["sessions"][0])
                encoded = json.dumps(payload)
                self.assertNotIn("super-secret-token", encoded)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/sessions/missing/stop",
                    data=b"",
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()


class StreamReadTests(unittest.TestCase):
    def test_collect_stream_returns_after_the_writer_goes_idle(self) -> None:
        import os
        import threading

        from kgate.kaggle_cli import collect_stream

        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "rb", buffering=0)

        def _write() -> None:
            os.write(write_fd, b"KGATE_BOOT {\"run\":\"abc\"}\n")
            time.sleep(0.8)
            os.close(write_fd)

        threading.Thread(target=_write, daemon=True).start()
        text = collect_stream(reader, idle_s=0.3, max_s=2)
        reader.close()
        self.assertIn("KGATE_BOOT", text)


class FrameTests(unittest.TestCase):
    def test_masked_roundtrip(self) -> None:
        left, right = socket.socketpair()
        try:
            left.sendall(encode_ws_frame(b"abc", opcode=0x2, mask=True))
            right.settimeout(2)
            opcode, data = read_ws_frame(right)
            self.assertEqual(opcode, 0x2)
            self.assertEqual(data, b"abc")
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
