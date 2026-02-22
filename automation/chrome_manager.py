from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class ChromeInstance:
    port: int
    ws_endpoint: str
    profile_dir: Path
    pid: int | None


class AutomationChromeManager:
    def __init__(self, extension_dir: Path, port: int = 9222, timeout_s: float = 20.0):
        self.extension_dir = extension_dir
        self.port = port
        self.timeout_s = timeout_s
        self.process: subprocess.Popen[str] | None = None

    def _detect_chrome_path(self) -> Path:
        candidates = [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Google Chrome not found in standard Windows locations")

    def _profile_dir(self) -> Path:
        appdata = os.getenv("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        profile_dir = base / "GrokAutomation" / "chrome-profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _fetch_json_version(self) -> dict[str, Any] | None:
        try:
            resp = requests.get(f"http://127.0.0.1:{self.port}/json/version", timeout=1.0)
            if resp.ok:
                return resp.json()
        except Exception:
            return None
        return None

    def launch_or_reuse(self) -> ChromeInstance:
        existing = self._wait_for_ready(ready_timeout_s=1.0)
        if existing is not None:
            return existing

        chrome_path = self._detect_chrome_path()
        profile_dir = self._profile_dir()
        args = [
            str(chrome_path),
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
            "--disable-features=Translate",
            f"--load-extension={self.extension_dir}",
            f"--disable-extensions-except={self.extension_dir}",
        ]
        self.process = subprocess.Popen(args)

        ready = self._wait_for_ready(ready_timeout_s=self.timeout_s)
        if ready is None:
            raise RuntimeError("Automation Chrome failed to become ready on /json/version")
        ready.pid = self.process.pid if self.process else None
        return ready

    def _wait_for_ready(self, ready_timeout_s: float) -> ChromeInstance | None:
        start = time.time()
        while time.time() - start <= ready_timeout_s:
            version = self._fetch_json_version()
            ws_endpoint = (version or {}).get("webSocketDebuggerUrl")
            if ws_endpoint:
                return ChromeInstance(
                    port=self.port,
                    ws_endpoint=str(ws_endpoint),
                    profile_dir=self._profile_dir(),
                    pid=self.process.pid if self.process else None,
                )
            time.sleep(0.25)
        return None
