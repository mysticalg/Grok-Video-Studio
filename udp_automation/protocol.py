from __future__ import annotations

import uuid
from typing import Any

VERSION = 1


def make_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def cmd(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"v": VERSION, "type": "cmd", "id": make_id(), "name": name, "payload": payload}


def event(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"v": VERSION, "type": "event", "id": make_id(), "name": name, "payload": payload}


def valid(msg: dict[str, Any]) -> bool:
    return isinstance(msg, dict) and all(k in msg for k in ("v", "type", "id"))
