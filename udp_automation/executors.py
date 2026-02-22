from __future__ import annotations

import json
import threading
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from udp_automation.protocol import cmd


ACTION_TIMEOUTS: dict[str, float] = {
    "platform.open": 25.0,
    "platform.ensure_logged_in": 20.0,
    "upload.select_file": 45.0,
    "form.fill": 35.0,
    "dom.click": 20.0,
    "dom.type": 25.0,
    "post.submit": 90.0,
    "post.status": 25.0,
}


class BaseExecutor:
    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddedExecutor(BaseExecutor):
    def __init__(self, handler: Callable[[str, dict[str, Any]], dict[str, Any]]):
        self.handler = handler

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.handler(action, payload)


class UdpExecutor(BaseExecutor):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18793,
        timeout_s: float = 60.0,
        retries: int = 2,
        stop_event: threading.Event | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.retries = retries
        self.stop_event = stop_event or threading.Event()
        self.log_path = self._resolve_log_path()

    def _resolve_log_path(self) -> Path:
        logs_dir = Path(__file__).resolve().parents[1] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / "udp_automation.log"

    def _log(self, action: str, status: str, detail: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] action={action} status={status} detail={detail}\n"
        try:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        except Exception:
            return

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stop_event.is_set():
            raise RuntimeError("UDP workflow stopped by user")
        message = cmd(action, payload)
        self._log(action, "start", f"id={message['id']}")
        for attempt in range(self.retries + 1):
            if self.stop_event.is_set():
                self._log(action, "stopped", f"attempt={attempt + 1}")
                raise RuntimeError("UDP workflow stopped by user")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                action_timeout_s = float(ACTION_TIMEOUTS.get(action, self.timeout_s))
                sock.settimeout(min(0.5, action_timeout_s))
                sock.sendto(json.dumps(message).encode("utf-8"), (self.host, self.port))
                deadline = time.time() + action_timeout_s
                self._log(action, "sent", f"attempt={attempt + 1} timeout={action_timeout_s:.1f}s")
                while time.time() < deadline:
                    if self.stop_event.is_set():
                        self._log(action, "stopped", f"attempt={attempt + 1}")
                        raise RuntimeError("UDP workflow stopped by user")
                    try:
                        data, _ = sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    response = json.loads(data.decode("utf-8"))
                    if response.get("type") == "event":
                        self._log(action, "event", json.dumps(response.get("payload") or {}, ensure_ascii=False))
                        continue
                    if response.get("id") == message["id"]:
                        if response.get("ok"):
                            self._log(action, "ok", json.dumps(response.get("payload") or {}, ensure_ascii=False))
                            return response
                        self._log(action, "error", str(response.get("error") or "command failed"))
                        raise RuntimeError(response.get("error") or f"Command failed: {action}")
            if attempt < self.retries:
                self._log(action, "retry", f"attempt={attempt + 1}")
                time.sleep(0.35)
        self._log(action, "timeout", f"retries={self.retries}")
        raise TimeoutError(f"No UDP cmd_ack for {action}")
