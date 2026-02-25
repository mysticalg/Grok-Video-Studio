from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from udp_automation.protocol import cmd


class BaseExecutor:
    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddedExecutor(BaseExecutor):
    def __init__(self, handler: Callable[[str, dict[str, Any]], dict[str, Any]]):
        self.handler = handler

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.handler(action, payload)


class UdpExecutor(BaseExecutor):
    ACTION_TIMEOUT_OVERRIDES_S = {
        # Rich-text editors (notably YouTube Studio) can take several seconds
        # to acknowledge synthetic input events before the extension replies.
        "form.fill": 90.0,
    }
    ACTION_TIMEOUT_PLATFORM_OVERRIDES_S = {
        # For YouTube, form.fill is best-effort; fail fast so workflow can continue.
        ("form.fill", "youtube"): 30.0,
    }
    ACTION_RETRY_OVERRIDES = {
        ("form.fill", "x"): 0,
        ("form.fill", "youtube"): 0,
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18793,
        timeout_s: float = 60.0,
        retries: int = 2,
        stop_event: threading.Event | None = None,
        action_delay_ms: int = 0,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.retries = retries
        self.stop_event = stop_event or threading.Event()
        self.action_delay_ms = max(0, int(action_delay_ms))
        self.log_path = self._resolve_log_path()
        self._run_lock = threading.Lock()
        self._action_counter = 0

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

    def _target_summary(self, payload: dict[str, Any]) -> str:
        target_fields = {
            "platform": payload.get("platform"),
            "selector": payload.get("selector"),
            "url": payload.get("url"),
            "filePath": payload.get("filePath"),
            "mode": payload.get("mode") or payload.get("publishMode"),
            "value": payload.get("value"),
        }
        fields = [f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in target_fields.items() if v not in (None, "")]

        fields_payload = payload.get("fields")
        if isinstance(fields_payload, dict) and fields_payload:
            field_keys = ",".join(sorted(str(k) for k in fields_payload.keys()))
            fields.append(f"fields={field_keys}")

        if not fields:
            return "target=<none>"
        return "target={" + ", ".join(fields) + "}"

    def _effective_retries(self, action: str, payload: dict[str, Any]) -> int:
        platform = str(payload.get("platform") or "").lower()
        override = self.ACTION_RETRY_OVERRIDES.get((action, platform))
        if override is None:
            return self.retries
        return max(0, int(override))

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._run_lock:
            self._action_counter += 1
            action_idx = self._action_counter

            if self.stop_event.is_set():
                raise RuntimeError("UDP workflow stopped by user")

            target_summary = self._target_summary(payload)
            platform = str(payload.get("platform") or "").lower()
            action_timeout_s = float(
                self.ACTION_TIMEOUT_PLATFORM_OVERRIDES_S.get(
                    (action, platform),
                    self.ACTION_TIMEOUT_OVERRIDES_S.get(action, self.timeout_s),
                )
            )
            action_retries = self._effective_retries(action, payload)
            total_attempts = action_retries + 1

            if action_idx > 1 and self.action_delay_ms > 0:
                delay_s = self.action_delay_ms / 1000.0
                self._log(action, "delay", f"sequence={action_idx} sleep_ms={self.action_delay_ms} {target_summary}")
                deadline = time.time() + delay_s
                while time.time() < deadline:
                    if self.stop_event.is_set():
                        self._log(action, "stopped", f"sequence={action_idx} during_delay=true {target_summary}")
                        raise RuntimeError("UDP workflow stopped by user")
                    time.sleep(min(0.1, max(0.01, deadline - time.time())))

            message = cmd(action, payload)
            self._log(
                action,
                "start",
                f"sequence={action_idx} id={message['id']} {target_summary} payload={json.dumps(payload, ensure_ascii=False)}",
            )

            for attempt in range(total_attempts):
                if self.stop_event.is_set():
                    self._log(action, "stopped", f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary}")
                    raise RuntimeError("UDP workflow stopped by user")

                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(min(0.5, action_timeout_s))
                    sock.sendto(json.dumps(message).encode("utf-8"), (self.host, self.port))
                    deadline = time.time() + action_timeout_s
                    self._log(action, "sent", f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary}")

                    while time.time() < deadline:
                        if self.stop_event.is_set():
                            self._log(action, "stopped", f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary}")
                            raise RuntimeError("UDP workflow stopped by user")
                        try:
                            data, _ = sock.recvfrom(65535)
                        except socket.timeout:
                            continue

                        response = json.loads(data.decode("utf-8"))
                        if response.get("type") == "event":
                            self._log(
                                action,
                                "event",
                                f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary} payload={json.dumps(response.get('payload') or {}, ensure_ascii=False)}",
                            )
                            continue

                        if response.get("id") == message["id"]:
                            if response.get("ok"):
                                self._log(
                                    action,
                                    "ok",
                                    f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary} payload={json.dumps(response.get('payload') or {}, ensure_ascii=False)}",
                                )
                                return response
                            self._log(
                                action,
                                "error",
                                f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary} error={str(response.get('error') or 'command failed')}",
                            )
                            raise RuntimeError(response.get("error") or f"Command failed: {action}")

                if attempt < action_retries:
                    self._log(action, "retry", f"sequence={action_idx} attempt={attempt + 1}/{total_attempts} {target_summary}")
                    time.sleep(0.35)

            self._log(action, "timeout", f"sequence={action_idx} retries={action_retries} {target_summary}")
            raise TimeoutError(f"No UDP cmd_ack for {action}")
