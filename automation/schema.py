#     *    
#    * *   
#   * * *  
#  *  *  * 
# *********
#  *  *  * 
#   * * *  
#    * *   
#     *    

"""Shared JSON message-schema helpers for the automation control plane."""

from __future__ import annotations

import uuid
from typing import Any

SCHEMA_VERSION = 1


def make_id() -> str:
    """Generate a unique identifier for each message crossing the control bus."""
    return f"msg_{uuid.uuid4().hex}"


def wrap_cmd(name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap command data with protocol metadata."""
    return {
        "v": SCHEMA_VERSION,
        "type": "cmd",
        "id": make_id(),
        "name": name,
        "payload": payload or {},
    }


def wrap_event(name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap asynchronous event data with protocol metadata."""
    return {
        "v": SCHEMA_VERSION,
        "type": "event",
        "id": make_id(),
        "name": name,
        "payload": payload or {},
    }


def validate(msg: dict[str, Any]) -> bool:
    """Return True when a payload has the required protocol keys."""
    if not isinstance(msg, dict):
        return False
    for key in ("v", "type", "id"):
        if key not in msg:
            return False
    return True
