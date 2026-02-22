from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from automation.cdp_controller import CDPController
from automation.chrome_manager import AutomationChromeManager
from automation.control_bus import ControlBusServer
from automation.schema import wrap_cmd
from udp_automation.protocol import event


PLATFORM_URLS = {
    "youtube": "https://studio.youtube.com",
    "tiktok": "https://www.tiktok.com/upload",
    "facebook": "https://www.facebook.com/reels/create",
}


class UdpAutomationService:
    def __init__(self, extension_dir: Path, host: str = "127.0.0.1", port: int = 18793, bus: ControlBusServer | None = None, start_bus: bool = True):
        self.host = host
        self.port = port
        self.extension_dir = extension_dir
        self.bus = bus or ControlBusServer()
        self._start_bus = start_bus
        self.chrome_manager = AutomationChromeManager(extension_dir=extension_dir)
        self.chrome_instance = None
        self.cdp: CDPController | None = None
        self._transport = None
        self._clients: set[tuple[str, int]] = set()

    async def start(self) -> None:
        if self._start_bus:
            await self.bus.start()
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(lambda: _UdpProtocol(self), local_addr=(self.host, self.port))

    async def stop(self) -> None:
        if self.cdp is not None:
            await self.cdp.close()
            self.cdp = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        if self._start_bus:
            await self.bus.stop()

    async def handle_command(self, msg: dict[str, Any], addr: tuple[str, int]) -> dict[str, Any]:
        self._clients.add(addr)
        name = msg.get("name")
        payload = msg.get("payload") or {}

        def _ack_from_extension(ack: dict[str, Any]) -> dict[str, Any]:
            response = {"ok": bool(ack.get("ok", False)), "payload": ack.get("payload", {})}
            error = ack.get("error")
            if error:
                response["error"] = error
            return response

        try:
            if name == "platform.open":
                platform = str(payload.get("platform") or "").lower()
                url = str(payload.get("url") or PLATFORM_URLS.get(platform) or "https://example.com")
                self.chrome_instance = self.chrome_manager.launch_or_reuse()
                if self.cdp is None:
                    self.cdp = await CDPController.connect(self.chrome_instance.ws_endpoint)
                page = await self.cdp.get_or_create_page(url)
                await page.bring_to_front()
                await self._emit("state", {"state": "page_opened", "platform": platform, "url": page.url})
                return {"ok": True, "payload": {"url": page.url}}

            if name == "upload.select_file":
                file_path = str(payload.get("filePath") or "")
                platform = str(payload.get("platform") or "")
                if not file_path:
                    raise RuntimeError("filePath is required")
                if self.cdp is None:
                    raise RuntimeError("CDP is not connected")
                page = await self.cdp.get_or_create_page(PLATFORM_URLS.get(platform.lower(), "https://example.com"))
                input_el = page.locator("input[type='file']").first
                try:
                    await input_el.wait_for(state="attached", timeout=15000)
                except Exception:
                    input_el = None
                if input_el is not None and await input_el.count() > 0:
                    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    try:
                        await input_el.set_input_files(file_path)
                        mode = "cdp_set_input_files"
                    except Exception as exc:
                        err_text = str(exc)
                        if "Cannot transfer files larger than 50Mb" in err_text:
                            used_dom_fallback = await self.cdp.set_file_input_files_via_dom(page, "input[type='file']", file_path)
                            if used_dom_fallback:
                                mode = "cdp_dom_set_file_input_files"
                            else:
                                await self._emit(
                                    "state",
                                    {
                                        "state": "upload_requires_manual_selection",
                                        "platform": platform,
                                        "reason": "remote_browser_file_transfer_limit",
                                        "fileSizeMb": round(file_size_mb, 2),
                                    },
                                )
                                return {
                                    "ok": True,
                                    "payload": {
                                        "requiresUserAction": True,
                                        "reason": "remote_browser_file_transfer_limit",
                                        "message": "Automatic upload over remote CDP failed for this file size",
                                        "fileSizeMb": round(file_size_mb, 2),
                                    },
                                }
                        else:
                            raise
                    await self._emit("state", {"state": "upload_selected", "platform": platform, "filePath": file_path, "mode": mode})
                    return {"ok": True, "payload": {"mode": mode, "fileSizeMb": round(file_size_mb, 2)}}
                ack = await self._send_extension_cmd("upload.select_file", payload)
                return _ack_from_extension(ack)

            if name in {"form.fill", "post.submit", "post.status", "dom.query", "dom.click", "dom.type", "platform.ensure_logged_in"}:
                ack = await self._send_extension_cmd(name, payload)
                return _ack_from_extension(ack)

            if name == "dom.ping":
                ack = await self._send_extension_cmd("dom.ping", payload)
                return _ack_from_extension(ack)

            raise RuntimeError(f"Unsupported command: {name}")
        except Exception as exc:
            return {"ok": False, "error": str(exc), "payload": {}}

    async def _send_extension_cmd(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        clients = list(self.bus.clients.keys())
        if not clients:
            raise RuntimeError("No extension client connected")
        return await self.bus.send_cmd(clients[0], wrap_cmd(name, payload))

    async def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self._transport is None:
            return
        message = json.dumps(event(name, payload)).encode("utf-8")
        for addr in self._clients:
            self._transport.sendto(message, addr)


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, service: UdpAutomationService):
        self.service = service
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        async def _handle() -> None:
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                return
            result = await self.service.handle_command(msg, addr)
            response = {
                "v": 1,
                "type": "cmd_ack",
                "id": msg.get("id"),
                "name": msg.get("name"),
                **result,
            }
            if self.transport is not None:
                self.transport.sendto(json.dumps(response).encode("utf-8"), addr)

        asyncio.create_task(_handle())
