"""On-disk accounts, sessions, and config. Tokens never go in the repo."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def kgate_home() -> Path:
    override = os.environ.get("KGATE_HOME")
    home = Path(override) if override else Path.home() / ".config" / "kgate"
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    return home


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class Store:
    def __init__(self, home: Path | None = None) -> None:
        self.home = home or kgate_home()
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        self.accounts_path = self.home / "accounts.json"
        self.sessions_path = self.home / "sessions.json"
        self.config_path = self.home / "config.json"
        self.token_dir = self.home / "tokens"
        self.kernel_dir = self.home / "kernels"
        self.log_dir = self.home / "logs"
        for directory in (self.token_dir, self.kernel_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)

    def config(self) -> dict[str, Any]:
        data = _read_json(self.config_path, {})
        data.setdefault("default_account", "")
        data.setdefault("accelerator", "NvidiaTeslaT4")
        data.setdefault("local_port", 8000)
        data.setdefault("dashboard_port", 8787)
        data.setdefault("kernel_slug", "kgate-session")
        return data

    def save_config(self, config: dict[str, Any]) -> None:
        _write_json(self.config_path, config)

    def list_accounts(self) -> list[dict[str, Any]]:
        data = _read_json(self.accounts_path, {"accounts": []})
        return list(data.get("accounts", []))

    def get_account(self, name: str) -> dict[str, Any]:
        for account in self.list_accounts():
            if account["name"] == name:
                return account
        raise KeyError(name)

    def default_account_name(self) -> str:
        config = self.config()
        if config.get("default_account"):
            return str(config["default_account"])
        accounts = self.list_accounts()
        if not accounts:
            raise KeyError("no accounts configured")
        return str(accounts[0]["name"])

    def resolve_account(self, name: str | None) -> dict[str, Any]:
        chosen = name or self.default_account_name()
        return self.get_account(chosen)

    def put_account(self, account: dict[str, Any]) -> None:
        accounts = [item for item in self.list_accounts() if item["name"] != account["name"]]
        accounts.append(account)
        accounts.sort(key=lambda item: item["name"])
        _write_json(self.accounts_path, {"accounts": accounts})

    def delete_account(self, name: str) -> None:
        accounts = [item for item in self.list_accounts() if item["name"] != name]
        _write_json(self.accounts_path, {"accounts": accounts})
        token_path = self.token_dir / name
        if token_path.exists():
            token_path.unlink()
        config = self.config()
        if config.get("default_account") == name:
            config["default_account"] = accounts[0]["name"] if accounts else ""
            self.save_config(config)

    def write_token(self, name: str, token: str) -> Path:
        path = self.token_dir / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token.strip() + "\n")
        return path

    def read_token(self, account: dict[str, Any]) -> str:
        path = Path(account["token_file"])
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"token file for {account['name']} is empty")
        return token

    def list_sessions(self) -> list[dict[str, Any]]:
        data = _read_json(self.sessions_path, {"sessions": []})
        return list(data.get("sessions", []))

    def get_session(self, session_id: str) -> dict[str, Any]:
        for session in self.list_sessions():
            if session["id"] == session_id:
                return session
        raise KeyError(session_id)

    def upsert_session(self, session: dict[str, Any]) -> None:
        sessions = [item for item in self.list_sessions() if item["id"] != session["id"]]
        sessions.append(session)
        sessions.sort(key=lambda item: item.get("started_at", ""))
        _write_json(self.sessions_path, {"sessions": sessions})

    def active_sessions(self, account: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for session in self.list_sessions():
            if session.get("status") in {"starting", "ready"}:
                if account is None or session.get("account") == account:
                    rows.append(session)
        return rows
