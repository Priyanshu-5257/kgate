"""kgate command line."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kgate import kaggle_cli
from kgate.control import ControlError, exec_command
from kgate.dashboard import ensure_dashboard, serve_dashboard
from kgate.launch import adopt_logs, describe, launch_session, refresh_session, stop_remote
from kgate.proxy import pick_port
from kgate.store import Store
from kgate.term import TermError, attach


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
ENGINES = ("none", "vllm", "ollama", "sglang")


def _store() -> Store:
    return Store()


def _die(message: str, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


def _account_or_die(store: Store, name: str | None) -> dict[str, Any] | int:
    try:
        return store.resolve_account(name)
    except KeyError:
        if name:
            return _die(f"No account named {name!r}. Add one with `kgate account add`.")
        return _die("No Kaggle account is configured. Run `kgate account add NAME --token-file ~/.kaggle/access_token`.")


def _add_account(store: Store, name: str, token: str) -> dict[str, Any]:
    if not NAME_RE.match(name):
        raise ValueError("account names use letters, numbers, '_' and '-', and start with a letter or number")
    pending = store.write_token(name + ".pending", token)
    try:
        username = kaggle_cli.whoami(token)
    except kaggle_cli.KaggleError:
        pending.unlink(missing_ok=True)
        raise
    final = store.write_token(name, pending.read_text(encoding="utf-8"))
    pending.unlink(missing_ok=True)
    account = {
        "name": name,
        "username": username,
        "token_file": str(final),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    store.put_account(account)
    config = store.config()
    if not config.get("default_account"):
        config["default_account"] = name
        store.save_config(config)
    return account


def _selected_session(store: Store, session_id: str | None, account: str | None) -> dict[str, Any]:
    if session_id:
        return store.get_session(session_id)
    rows = [row for row in store.list_sessions() if row.get("status") in {"starting", "ready"}]
    if account:
        rows = [row for row in rows if row.get("account") == account]
    if not rows:
        raise KeyError("no running session")
    return rows[-1]


def _print_session(session: dict[str, Any]) -> None:
    shown = describe(session)
    print(f"session {shown['id']}  {shown.get('status')}  {shown.get('kernel_url')}")
    if shown.get("base_url"):
        print(f"base URL {shown['base_url']}")
    if shown.get("model"):
        print(f"model {shown['model']} via {shown.get('engine')}")
    print("shell `kgate attach`    stop `kgate down`    watch `kgate dash`")


def cmd_account_add(args: argparse.Namespace) -> int:
    store = _store()
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    else:
        token = getpass.getpass("Kaggle access token: ").strip()
    if not token:
        return _die("The token is empty.")
    try:
        account = _add_account(store, args.name, token)
    except (kaggle_cli.KaggleError, OSError, ValueError) as exc:
        return _die(str(exc))
    print(f"Added {account['name']} ({account['username']}).")
    return 0


def cmd_account_list(_args: argparse.Namespace) -> int:
    store = _store()
    try:
        default = store.default_account_name()
    except KeyError:
        default = ""
    accounts = store.list_accounts()
    if not accounts:
        print("No accounts. `kgate account add NAME --token-file ~/.kaggle/access_token`")
        return 0
    for account in accounts:
        mark = " default" if account["name"] == default else ""
        print(f"{account['name']:16} {account.get('username', ''):20}{mark}")
    return 0


def cmd_account_use(args: argparse.Namespace) -> int:
    store = _store()
    try:
        store.get_account(args.name)
    except KeyError:
        return _die(f"No account named {args.name!r}.")
    config = store.config()
    config["default_account"] = args.name
    store.save_config(config)
    print(f"Default account is {args.name}.")
    return 0


def cmd_account_remove(args: argparse.Namespace) -> int:
    store = _store()
    try:
        store.get_account(args.name)
    except KeyError:
        return _die(f"No account named {args.name!r}.")
    active = [row for row in store.active_sessions(args.name)]
    if active and not args.force:
        return _die(f"{args.name} has a running session ({active[-1]['id']}). Stop it with `kgate down` first.")
    store.delete_account(args.name)
    print(f"Removed {args.name}.")
    return 0


def _print_quota(name: str, username: str, rows: list[dict[str, Any]]) -> None:
    print(f"{name} ({username})")
    for row in rows:
        resource = row.get("resource", "")
        used = row.get("used", "")
        remaining = row.get("remaining", "")
        total = row.get("total", "")
        refresh = row.get("refreshAt", "")
        print(f"  {resource:4}  left {remaining:>8}  used {used:>8}  of {total:>8}  resets {refresh}")


def cmd_quota(args: argparse.Namespace) -> int:
    store = _store()
    if args.account:
        try:
            accounts = [store.get_account(args.account)]
        except KeyError:
            return _die(f"No account named {args.account!r}.")
    else:
        accounts = store.list_accounts()
    if not accounts:
        return _die("No accounts configured.")
    failed = 0
    for account in accounts:
        try:
            rows = kaggle_cli.quota(store.read_token(account))
        except (kaggle_cli.KaggleError, OSError) as exc:
            print(f"{account['name']}: {exc}", file=sys.stderr)
            failed += 1
            continue
        _print_quota(account["name"], account.get("username", ""), rows)
    return 1 if failed == len(accounts) else 0


def cmd_up(args: argparse.Namespace) -> int:
    store = _store()
    resolved = _account_or_die(store, args.account)
    if isinstance(resolved, int):
        return resolved
    engine = args.engine
    if engine is None:
        engine = "vllm" if args.model else "none"
    if engine not in ENGINES:
        return _die(f"engine must be one of: {', '.join(ENGINES)}")
    hf_token = ""
    if args.hf_token_file:
        hf_token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()
    config = store.config()
    dash_url = ensure_dashboard(store)
    print(f"Dashboard {dash_url}", flush=True)
    try:
        session = launch_session(
            store,
            resolved,
            engine=engine,
            model=args.model or "",
            accelerator=args.accelerator or config["accelerator"],
            hours=args.hours,
            engine_args=args.engine_arg or [],
            max_model_len=args.max_model_len,
            tensor_parallel=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization=args.quantization or "",
            hf_token=hf_token,
            dtype=args.dtype or "",
            local_port=args.port or int(config["local_port"]),
            wait=not args.no_wait,
            slug=str(config["kernel_slug"]),
        )
    except (kaggle_cli.KaggleError, ValueError, RuntimeError, TimeoutError, OSError) as exc:
        return _die(str(exc))
    _print_session(session)
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    store = _store()
    try:
        session = _selected_session(store, args.session, args.account)
    except KeyError as exc:
        return _die(str(exc))
    try:
        account = store.get_account(str(session["account"]))
        note = stop_remote(store, session, store.read_token(account))
    except (KeyError, kaggle_cli.KaggleError, OSError) as exc:
        return _die(str(exc))
    print(note)
    return 0


def cmd_ps(args: argparse.Namespace) -> int:
    store = _store()
    rows = store.list_sessions()
    if args.account:
        rows = [row for row in rows if row.get("account") == args.account]
    rows = [refresh_session(store, row) if row.get("status") in {"starting", "ready"} else row for row in rows]
    if not rows:
        print("No sessions yet.")
        return 0
    for session in rows:
        shown = describe(session)
        model = shown.get("model") or "-"
        print(
            f"{shown['id']:24} {shown.get('status', ''):10} {shown.get('account', ''):12} "
            f"{shown.get('engine', ''):8} {model:32} {shown.get('base_url', '')}"
        )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    store = _store()
    session: dict[str, Any] | None
    if args.kernel:
        session = None
    else:
        try:
            session = _selected_session(store, args.session, args.account)
        except KeyError:
            rows = store.list_sessions()
            if args.account:
                rows = [row for row in rows if row.get("account") == args.account]
            if args.session or not rows:
                return _die("No session to read logs from. Pass --session or start one with `kgate up`.")
            session = rows[-1]
    if session is None:
        try:
            account = store.resolve_account(args.account)
        except KeyError as exc:
            return _die(str(exc))
        kernel = args.kernel
        token = store.read_token(account)
    else:
        kernel = str(session["kernel"])
        token = store.read_token(store.get_account(str(session["account"])))
    try:
        if not args.follow:
            logs = kaggle_cli.kernel_logs(token, kernel)
            if session is not None:
                adopt_logs(store, session, logs)
            print(logs, end="" if logs.endswith("\n") or not logs else "\n")
            return 0
        for chunk in kaggle_cli.follow_kernel_logs(token, kernel):
            print(chunk, end="", flush=True)
            if session is not None and "KGATE_TUNNEL" in chunk:
                adopt_logs(store, session, chunk)
    except KeyboardInterrupt:
        print()
        return 0
    except (kaggle_cli.KaggleError, KeyError, OSError) as exc:
        return _die(str(exc))


def cmd_attach(args: argparse.Namespace) -> int:
    store = _store()
    try:
        session = _selected_session(store, args.session, args.account)
    except KeyError as exc:
        return _die(str(exc))
    if not session.get("tunnel_url"):
        return _die("That session has no tunnel yet. Run `kgate logs` and wait for KGATE_TUNNEL.")
    try:
        return attach(str(session["tunnel_url"]), str(session["token"]))
    except TermError as exc:
        return _die(str(exc))


def cmd_exec(args: argparse.Namespace) -> int:
    store = _store()
    try:
        session = _selected_session(store, args.session, args.account)
    except KeyError as exc:
        return _die(str(exc))
    if not session.get("tunnel_url"):
        return _die("That session has no tunnel yet.")
    command = " ".join(args.command)
    try:
        result = exec_command(str(session["tunnel_url"]), str(session["token"]), command, timeout=args.timeout)
    except ControlError as exc:
        return _die(str(exc))
    sys.stdout.write(result.get("stdout") or "")
    sys.stderr.write(result.get("stderr") or "")
    return int(result.get("code") or 0)


def cmd_dash(args: argparse.Namespace) -> int:
    store = _store()
    port = args.port or pick_port(int(store.config()["dashboard_port"]))
    serve_dashboard(store, port)
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    store = _store()
    print(f"home {store.home}")
    try:
        binary = kaggle_cli.kaggle_bin()
    except kaggle_cli.KaggleError as exc:
        print(exc)
        return 1
    import subprocess

    version = subprocess.run([binary, "--version"], text=True, capture_output=True, check=False)
    text = ((version.stdout or "") + (version.stderr or "")).strip()
    print(f"kaggle {binary}")
    print(text.splitlines()[0] if text else "kaggle --version produced no output")
    accounts = store.list_accounts()
    print(f"accounts {len(accounts)}")
    if not accounts:
        print("Add an account with `kgate account add NAME --token-file ~/.kaggle/access_token`.")
        return 0
    failed = 0
    for account in accounts:
        try:
            rows = kaggle_cli.quota(store.read_token(account))
            gpu = next((row for row in rows if str(row.get("resource")).upper() == "GPU"), None)
            left = gpu.get("remaining") if gpu else "unknown"
            print(f"  {account['name']} ({account.get('username')}) GPU left {left}")
        except (kaggle_cli.KaggleError, OSError) as exc:
            print(f"  {account['name']}: {exc}")
            failed += 1
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kgate",
        description=(
            "Use your Kaggle GPU quota from this laptop. "
            "Each account gets a private notebook with a shell and, if you ask, "
            "vLLM, Ollama, or SGLang behind http://127.0.0.1:8000/v1."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    account = sub.add_parser("account", help="save and switch Kaggle accounts")
    account_sub = account.add_subparsers(dest="account_command", required=True)
    add = account_sub.add_parser("add", help="store a token and check that it works")
    add.add_argument("name")
    add.add_argument("--token-file", help="file that contains a Kaggle access token")
    add.set_defaults(func=cmd_account_add)
    listing = account_sub.add_parser("list", help="list configured accounts")
    listing.set_defaults(func=cmd_account_list)
    use = account_sub.add_parser("use", help="set the default account")
    use.add_argument("name")
    use.set_defaults(func=cmd_account_use)
    remove = account_sub.add_parser("remove", help="forget an account and delete its token file")
    remove.add_argument("name")
    remove.add_argument("--force", action="store_true")
    remove.set_defaults(func=cmd_account_remove)

    quota = sub.add_parser("quota", help="show GPU and TPU hours left on each account")
    quota.add_argument("--account")
    quota.set_defaults(func=cmd_quota)

    up = sub.add_parser("up", help="start a private GPU session and forward it to localhost")
    up.add_argument("--account")
    up.add_argument("--model", help="model id, for example Qwen/Qwen2.5-1.5B-Instruct or qwen2.5:7b")
    up.add_argument("--engine", choices=ENGINES, help="default is vllm when --model is set, otherwise a shell")
    up.add_argument("--accelerator", help="Kaggle machine shape, default NvidiaTeslaT4")
    up.add_argument("--hours", type=float, default=6, help="cap the session, also capped by remaining GPU quota")
    up.add_argument("--port", type=int, help="local port for the OpenAI API, default 8000")
    up.add_argument("--max-model-len", type=int, default=4096)
    up.add_argument("--tensor-parallel", type=int, default=1, help="0 uses every visible GPU")
    up.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    up.add_argument("--quantization", help="passed to vLLM, for example awq or gptq")
    up.add_argument("--dtype", help="default half on T4/P100, bfloat16 on A100/L4")
    up.add_argument("--engine-arg", action="append", help="extra argument for the engine, repeatable")
    up.add_argument("--hf-token-file", help="Hugging Face token for gated models. It is written into the private kernel.")
    up.add_argument("--no-wait", action="store_true", help="return as soon as the kernel is pushed")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="stop the session and the local proxy")
    down.add_argument("--session")
    down.add_argument("--account")
    down.set_defaults(func=cmd_down)

    ps = sub.add_parser("ps", help="list sessions started from this laptop")
    ps.add_argument("--account")
    ps.set_defaults(func=cmd_ps)

    logs = sub.add_parser("logs", help="show the Kaggle kernel log")
    logs.add_argument("--session")
    logs.add_argument("--account")
    logs.add_argument("--kernel", help="owner/slug, if you do not want the latest session")
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    shell = sub.add_parser("attach", help="open the session shell. Ctrl-] detaches")
    shell.add_argument("--session")
    shell.add_argument("--account")
    shell.set_defaults(func=cmd_attach)

    execute = sub.add_parser("exec", help="run one command on the session")
    execute.add_argument("--session")
    execute.add_argument("--account")
    execute.add_argument("--timeout", type=float, default=30)
    execute.add_argument("command", nargs=argparse.REMAINDER)
    execute.set_defaults(func=cmd_exec)

    dash = sub.add_parser("dash", help="open the local usage dashboard")
    dash.add_argument("--port", type=int)
    dash.set_defaults(func=cmd_dash)

    doctor = sub.add_parser("doctor", help="check the Kaggle CLI, accounts, and quota")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", "") == "exec" and not args.command:
        return _die("pass a command, for example `kgate exec nvidia-smi`")
    if getattr(args, "command", "") == "logs" and args.kernel and args.session:
        return _die("pass either --session or --kernel")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
