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
    "instagram": "https://www.instagram.com/create/reel/",
    "x": "https://x.com/compose/post",
}


EXTENSION_CMD_TIMEOUTS = {
    "form.fill": 90.0,
    "post.submit": 150.0,
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

    @staticmethod
    def _is_connection_closed_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "connection closed" in text or "target page, context or browser has been closed" in text

    async def _connect_cdp(self) -> None:
        if self.chrome_instance is None:
            raise RuntimeError("Chrome instance is not available")
        if self.cdp is not None:
            try:
                await self.cdp.close()
            except Exception:
                # The previous CDP connection may already be gone (e.g. driver closed).
                # Swallow teardown failures and establish a fresh session.
                pass
            finally:
                self.cdp = None
        self.cdp = await CDPController.connect(self.chrome_instance.ws_endpoint)

    async def _open_platform_page(self, url: str, reuse_tab: bool) -> Any:
        if self.cdp is None:
            await self._connect_cdp()
        try:
            page = await self.cdp.get_or_create_page(url, reuse_tab=reuse_tab)
            await page.bring_to_front()
            return page
        except Exception as exc:
            if not self._is_connection_closed_error(exc):
                raise

        # The original browser/debugger session may have been torn down.
        # Re-attach to a running browser (or relaunch) and retry once.
        self.chrome_instance = self.chrome_manager.launch_or_reuse()
        await self._connect_cdp()
        page = await self.cdp.get_or_create_page(url, reuse_tab=reuse_tab)
        await page.bring_to_front()
        return page

    async def start(self) -> None:
        if self._start_bus:
            await self.bus.start()
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(lambda: _UdpProtocol(self), local_addr=(self.host, self.port))

    async def stop(self) -> None:
        if self.cdp is not None:
            try:
                await self.cdp.close()
            finally:
                self.cdp = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        if self._start_bus:
            await self.bus.stop()


    async def _fill_youtube_fields_via_cdp(self, fields: dict[str, Any]) -> dict[str, bool] | None:
        if self.cdp is None:
            await self._connect_cdp()
        if self.cdp is None:
            return None

        page = await self.cdp.find_page_by_url_contains("studio.youtube.com")
        if page is None:
            page = await self.cdp.get_most_recent_page()
        if page is None:
            return None

        title = str((fields or {}).get("title") or "")
        description = str((fields or {}).get("description") or "")
        if not title and not description:
            return {}

        async def _set_rich_text(locator, value: str) -> bool:
            text = str(value or "")
            if await locator.count() == 0:
                return False
            target = locator.first
            try:
                await target.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass

            script = """
(el, value) => {
  const normalize = (text) => String(text || '').replace(/\\u200B/g, '').replace(/\\s+/g, ' ').trim();
  const expected = normalize(value);
  try { el.focus(); } catch (_) {}
  let inserted = false;
  try {
    const sel = window.getSelection?.();
    const range = document.createRange();
    range.selectNodeContents(el);
    range.collapse(false);
    sel?.removeAllRanges();
    sel?.addRange(range);
  } catch (_) {}
  try {
    document.execCommand('selectAll', false, null);
    document.execCommand('delete', false, null);
    inserted = Boolean(document.execCommand('insertText', false, String(value || '')));
  } catch (_) {
    inserted = false;
  }
  if (!inserted) {
    try { el.textContent = String(value || ''); } catch (_) {}
  }
  try { el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, composed: true, data: String(value || ''), inputType: 'insertText' })); } catch (_) {}
  try { el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: String(value || ''), inputType: 'insertText' })); } catch (_) {
    try { el.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
  }
  try { el.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
  const current = normalize(el?.innerText || el?.textContent || '');
  if (!expected) return current.length === 0;
  return current === expected || current.includes(expected) || expected.includes(current);
}
"""
            try:
                return bool(await target.evaluate(script, text))
            except Exception:
                return False

        title_locator = page.locator(
            "#title-textarea #textbox[contenteditable='true'], "
            "div#textbox[contenteditable='true'][aria-label*='Add a title' i], "
            "div#textbox[contenteditable='true'][aria-label*='title' i][aria-required='true']"
        )
        description_locator = page.locator(
            "#description #textbox[contenteditable='true'], "
            "div#textbox[contenteditable='true'][aria-label*='Tell viewers about your video' i], "
            "div#textbox[contenteditable='true'][aria-label*='description' i][aria-required='false']"
        )

        async def _run() -> dict[str, bool]:
            result = {"title": False, "description": False}
            for _ in range(6):
                if title:
                    result["title"] = await _set_rich_text(title_locator, title)
                else:
                    result["title"] = True

                if description:
                    result["description"] = await _set_rich_text(description_locator, description)
                else:
                    result["description"] = True

                if result["title"] and result["description"]:
                    return result
                try:
                    await page.wait_for_timeout(250)
                except Exception:
                    pass
            return result

        try:
            return await asyncio.wait_for(_run(), timeout=8.0)
        except Exception:
            return {"title": False if title else True, "description": False if description else True}

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
                reuse_tab = bool(payload.get("reuseTab", False))
                page = await self._open_platform_page(url, reuse_tab=reuse_tab)
                await self._emit("state", {"state": "page_opened", "platform": platform, "url": page.url})
                return {"ok": True, "payload": {"url": page.url}}

            if name == "upload.select_file":
                file_path = str(payload.get("filePath") or "")
                platform = str(payload.get("platform") or "")
                if not file_path:
                    raise RuntimeError("filePath is required")
                if self.cdp is None:
                    raise RuntimeError("CDP is not connected")
                page = await self.cdp.get_most_recent_page()
                if page is None:
                    page = await self.cdp.get_or_create_page(
                        PLATFORM_URLS.get(platform.lower(), "https://example.com"),
                        reuse_tab=True,
                    )
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
                        extension_used = False
                        # For remote CDP file-transfer limits, use extension-side debugger injection on the browser host.
                        if "Cannot transfer files larger than 50Mb" in err_text:
                            try:
                                ack = await self._send_extension_cmd("upload.select_file", payload)
                                ack_payload = _ack_from_extension(ack)
                                if bool(ack_payload.get("ok")):
                                    mode = str((ack_payload.get("payload") or {}).get("mode") or "extension_debugger_set_file_input_files")
                                    extension_used = True
                            except Exception:
                                extension_used = False

                        if not extension_used:
                            used_dom_fallback = False
                            if self.cdp is not None:
                                for _ in range(3):
                                    used_dom_fallback = await self.cdp.set_file_input_files_via_dom(page, "input[type='file']", file_path)
                                    if used_dom_fallback:
                                        break
                                    try:
                                        await page.wait_for_timeout(250)
                                    except Exception:
                                        pass
                            if used_dom_fallback:
                                mode = "cdp_dom_set_file_input_files"
                            else:
                                raise RuntimeError(
                                    "Automatic upload over remote CDP failed to set the file input"
                                    f" (sizeMb={round(file_size_mb, 2)}): {err_text}"
                                )
                    await self._emit("state", {"state": "upload_selected", "platform": platform, "filePath": file_path, "mode": mode})
                    return {"ok": True, "payload": {"mode": mode, "fileSizeMb": round(file_size_mb, 2)}}
                ack = await self._send_extension_cmd("upload.select_file", payload)
                return _ack_from_extension(ack)

            if name == "platform.ensure_logged_in":
                platform = str(payload.get("platform") or "")
                await self._emit("state", {"state": "login_check_skipped", "platform": platform, "loggedIn": True})
                return {"ok": True, "payload": {"platform": platform, "loggedIn": True, "mode": "service_fastpath"}}

            if name == "form.fill":
                platform = str(payload.get("platform") or "").lower()
                fields = payload.get("fields") or {}
                if platform == "youtube" and isinstance(fields, dict):
                    cdp_result = await self._fill_youtube_fields_via_cdp(fields)
                    if isinstance(cdp_result, dict) and cdp_result:
                        return {"ok": True, "payload": cdp_result}
                ack = await self._send_extension_cmd(name, payload)
                return _ack_from_extension(ack)

            if name in {"post.submit", "post.status", "dom.query", "dom.click", "dom.type"}:
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

        timeout_s = float(EXTENSION_CMD_TIMEOUTS.get(name, 15.0))
        last_error: Exception | None = None
        for client_id in clients:
            try:
                return await self.bus.send_cmd(client_id, wrap_cmd(name, payload), timeout_s=timeout_s)
            except Exception as exc:
                last_error = exc
                self.bus.clients.pop(client_id, None)
                continue
        raise RuntimeError(str(last_error) if last_error else "No extension client connected")

    async def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self._transport is None:
            return
        message = json.dumps(event(name, payload)).encode("utf-8")
        dead_clients: list[tuple[str, int]] = []
        for addr in self._clients:
            try:
                self._transport.sendto(message, addr)
            except OSError:
                dead_clients.append(addr)
        for addr in dead_clients:
            self._clients.discard(addr)


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

            try:
                result = await self.service.handle_command(msg, addr)
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "payload": {}}

            response = {
                "v": 1,
                "type": "cmd_ack",
                "id": msg.get("id"),
                "name": msg.get("name"),
                **result,
            }
            if self.transport is not None:
                try:
                    self.transport.sendto(json.dumps(response).encode("utf-8"), addr)
                except OSError:
                    return

        asyncio.create_task(_handle())
