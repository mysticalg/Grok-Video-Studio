from __future__ import annotations

import json
import socket
import time
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
    def __init__(self, host: str = "127.0.0.1", port: int = 18793, timeout_s: float = 20.0, retries: int = 2):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.retries = retries

    def run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        message = cmd(action, payload)
        for attempt in range(self.retries + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout_s)
                sock.sendto(json.dumps(message).encode("utf-8"), (self.host, self.port))
                deadline = time.time() + self.timeout_s
                while time.time() < deadline:
                    data, _ = sock.recvfrom(65535)
                    response = json.loads(data.decode("utf-8"))
                    if response.get("type") == "event":
                        continue
                    if response.get("id") == message["id"]:
                        if response.get("ok"):
                            return response
                        raise RuntimeError(response.get("error") or f"Command failed: {action}")
            if attempt < self.retries:
                time.sleep(0.35)
        raise TimeoutError(f"No UDP cmd_ack for {action}")
