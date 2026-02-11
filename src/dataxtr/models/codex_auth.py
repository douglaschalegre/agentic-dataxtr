"""Helpers for resolving OpenAI Codex OAuth credentials."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"


@dataclass
class CodexAuth:
    """Resolved auth payload for OpenAI Codex."""

    access_token: str
    account_id: Optional[str] = None


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"Codex auth file not found at {path}. Run `codex login` first "
            "or set OPENAI_CODEX_ACCESS_TOKEN."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Codex auth file is not valid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Codex auth file has invalid structure: {path}")
    return data


def resolve_codex_auth() -> CodexAuth:
    """Resolve Codex OAuth token from environment or Codex auth.json.

    Resolution order:
    1) OPENAI_CODEX_ACCESS_TOKEN (+ optional OPENAI_CODEX_ACCOUNT_ID)
    2) OPENAI_CODEX_AUTH_FILE (or ~/.codex/auth.json)
    """

    env_token = os.getenv("OPENAI_CODEX_ACCESS_TOKEN", "").strip()
    if env_token:
        env_account_id = os.getenv("OPENAI_CODEX_ACCOUNT_ID", "").strip() or None
        return CodexAuth(access_token=env_token, account_id=env_account_id)

    auth_path = Path(os.getenv("OPENAI_CODEX_AUTH_FILE", str(DEFAULT_CODEX_AUTH_FILE))).expanduser()
    payload = _read_json_file(auth_path)

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError(
            "Codex auth file does not contain tokens. Run `codex login` "
            "or set OPENAI_CODEX_ACCESS_TOKEN."
        )

    access_token = str(tokens.get("access_token", "")).strip()
    if not access_token:
        raise ValueError(
            "Codex auth file is missing tokens.access_token. Run `codex login` "
            "or set OPENAI_CODEX_ACCESS_TOKEN."
        )

    account_id = str(tokens.get("account_id", "")).strip() or None
    return CodexAuth(access_token=access_token, account_id=account_id)
