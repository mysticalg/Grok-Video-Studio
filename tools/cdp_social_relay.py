#!/usr/bin/env python3
"""Minimal local CDP relay endpoint for Grok Video Studio social uploads.

This relay is intentionally simple:
- Exposes POST /social-upload-step
- Logs incoming payloads
- Returns `handled: false` by default so app falls back to built-in DOM automation

Use this to eliminate connection-refused errors and validate relay wiring first,
then extend handler logic to drive CDP actions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class RelayHandler(BaseHTTPRequestHandler):
    server_version = "GrokCDPRelay/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {self.client_address[0]} - {fmt % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/social-upload-step":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        platform = str(payload.get("platform") or "unknown")
        attempt = int(payload.get("attempt") or 0)
        current_url = str(payload.get("current_url") or "")
        debug_port = str(payload.get("qtwebengine_remote_debugging") or "")

        print(
            f"relay step: platform={platform} attempt={attempt} "
            f"debug_port={debug_port or '(unset)'} url={current_url}"
        )

        # Starter behavior: acknowledge request but let app use built-in DOM automation.
        self._send_json(
            200,
            {
                "handled": False,
                "done": False,
                "status": "Relay reachable; using in-app DOM fallback.",
                "progress": 35,
                "retry_ms": 1200,
                "log": "Stub relay active. Implement CDP actions in tools/cdp_social_relay.py.",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CDP relay for social upload steps.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RelayHandler)
    print(f"CDP relay listening on http://{args.host}:{args.port}/social-upload-step")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
