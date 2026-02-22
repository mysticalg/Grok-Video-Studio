#!/usr/bin/env python3
"""Chrome Native Messaging host for Grok Video Studio.

Protocol:
- Reads length-prefixed JSON messages from stdin.
- Writes length-prefixed JSON responses to stdout.

Supported request schema:
{
  "id": "optional-correlation-id",
  "action": "ping" | "social_upload_step" | "http_proxy",
  ...
}
"""

from __future__ import annotations

import json
import os
import struct
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_RELAY_URL = os.environ.get(
    "GROK_NATIVE_HOST_RELAY_URL", "http://127.0.0.1:8765/social-upload-step"
)


def _read_message() -> dict[str, Any] | None:
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    if len(raw_length) < 4:
        raise EOFError("Invalid native message length prefix")
    message_length = struct.unpack("=I", raw_length)[0]
    message = sys.stdin.buffer.read(message_length)
    if len(message) < message_length:
        raise EOFError("Unexpected EOF while reading native message")
    return json.loads(message.decode("utf-8"))


def _send_message(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _http_json(
    url: str,
    method: str = "POST",
    body: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        raw = response.read()
        text = raw.decode("utf-8") if raw else "{}"
        return json.loads(text)


def _handle_request(msg: dict[str, Any]) -> dict[str, Any]:
    req_id = msg.get("id")
    action = msg.get("action", "ping")

    if action == "ping":
        return {
            "id": req_id,
            "ok": True,
            "action": "ping",
            "app": "grok-video-studio-native-host",
        }

    if action == "social_upload_step":
        relay_url = msg.get("relay_url") or DEFAULT_RELAY_URL
        relay_payload = msg.get("payload") or {}
        relay_result = _http_json(relay_url, body=relay_payload)
        return {
            "id": req_id,
            "ok": True,
            "action": action,
            "relay_url": relay_url,
            "result": relay_result,
        }

    if action == "http_proxy":
        url = msg.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("'url' is required for action=http_proxy")
        method = str(msg.get("method", "POST")).upper()
        body = msg.get("body") if isinstance(msg.get("body"), dict) else {}
        result = _http_json(url=url, method=method, body=body)
        return {
            "id": req_id,
            "ok": True,
            "action": action,
            "url": url,
            "result": result,
        }

    raise ValueError(f"Unknown action: {action}")


def main() -> int:
    try:
        while True:
            incoming = _read_message()
            if incoming is None:
                break
            try:
                response = _handle_request(incoming)
            except urllib.error.URLError as exc:
                response = {
                    "id": incoming.get("id"),
                    "ok": False,
                    "error": f"Network error: {exc}",
                }
            except Exception as exc:  # noqa: BLE001
                response = {
                    "id": incoming.get("id"),
                    "ok": False,
                    "error": str(exc),
                }
            _send_message(response)
        return 0
    except Exception as exc:  # noqa: BLE001
        try:
            _send_message({"ok": False, "fatal": True, "error": str(exc)})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
