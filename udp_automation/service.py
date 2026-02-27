from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
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
    "instagram": "https://www.instagram.com/",
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

        script = """
(value) => {
  const normalize = (text) => String(text || '').replace(/\\u200B/g, '').replace(/\\s+/g, ' ').trim();
  const byRoleTextbox = Array.from(document.querySelectorAll("div#textbox[contenteditable='true'][role='textbox']"));

  const findField = (key) => {
    if (key === 'title') {
      return byRoleTextbox.find((el) => /add a title|describes your video|title/i.test(String(el.getAttribute('aria-label') || '')))
        || document.querySelector("#title-textarea #textbox[contenteditable='true']")
        || null;
    }
    return byRoleTextbox.find((el) => /tell viewers about your video|description/i.test(String(el.getAttribute('aria-label') || '')))
      || document.querySelector("#description #textbox[contenteditable='true']")
      || null;
  };

  const setField = (el, text) => {
    if (!el) return false;
    const val = String(text || '');
    try { el.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
    try { el.focus(); } catch (_) {}

    let applied = false;
    try {
      const sel = window.getSelection?.();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      sel?.removeAllRanges();
      sel?.addRange(range);
      document.execCommand('selectAll', false, null);
      document.execCommand('delete', false, null);
      applied = Boolean(document.execCommand('insertText', false, val));
    } catch (_) {
      applied = false;
    }

    if (!applied) {
      try { el.textContent = val; } catch (_) {}
    }

    try { el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, composed: true, data: val, inputType: 'insertText' })); } catch (_) {}
    try { el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: val, inputType: 'insertText' })); } catch (_) {
      try { el.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
    }
    try { el.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
    try { el.dispatchEvent(new Event('blur', { bubbles: true })); } catch (_) {}

    const expected = normalize(val);
    const current = normalize(el.innerText || el.textContent || '');
    if (!expected) return current.length === 0;
    return current === expected || current.includes(expected) || expected.includes(current);
  };

  const out = { title: true, description: true };
  if (Object.prototype.hasOwnProperty.call(value, 'title')) {
    out.title = setField(findField('title'), value.title);
  }
  if (Object.prototype.hasOwnProperty.call(value, 'description')) {
    out.description = setField(findField('description'), value.description);
  }
  return out;
}
"""

        async def _run() -> dict[str, bool]:
            result = {"title": not bool(title), "description": not bool(description)}
            for _ in range(6):
                try:
                    payload = await page.evaluate(script, {"title": title, "description": description})
                    if isinstance(payload, dict):
                        result["title"] = bool(payload.get("title", result["title"]))
                        result["description"] = bool(payload.get("description", result["description"]))
                except Exception:
                    pass

                if result["title"] and result["description"]:
                    return result
                try:
                    await page.wait_for_timeout(250)
                except Exception:
                    pass
            return result

        try:
            return await asyncio.wait_for(_run(), timeout=6.0)
        except Exception:
            return {"title": False if title else True, "description": False if description else True}


    def _openai_fill_plan(self, html_fragment: str, title: str, description: str, timeout_s: float) -> dict[str, list[str]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return {"title": [], "description": []}

        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are analyzing YouTube Studio upload dialog HTML. Return strict JSON only: "
                        "{\"title\":[selector...],\"description\":[selector...]} with best CSS selectors "
                        "for title and description contenteditable fields."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Title value: {title}\nDescription value: {description}\n"
                        f"HTML:\n{html_fragment[:120000]}"
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}")
            parsed = json.loads(content)
            title_sels = [str(x) for x in (parsed.get("title") or []) if str(x).strip()]
            desc_sels = [str(x) for x in (parsed.get("description") or []) if str(x).strip()]
            return {"title": title_sels[:12], "description": desc_sels[:12]}
        except Exception:
            return {"title": [], "description": []}

    async def _fill_youtube_fields_via_openai(self, fields: dict[str, Any]) -> dict[str, bool] | None:
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

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            await self._emit("state", {"state": "youtube_openai_skipped", "reason": "missing_openai_api_key"})
            return None

        await self._emit("state", {"state": "youtube_openai_plan_start"})
        html_fragment = await page.evaluate(
            """() => {
                const dialog = document.querySelector('ytcp-uploads-dialog');
                const root = dialog || document.body;
                return String(root?.outerHTML || document.body?.outerHTML || '').slice(0, 200000);
            }"""
        )
        openai_timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "6"))
        plan = await asyncio.to_thread(self._openai_fill_plan, str(html_fragment or ""), title, description, openai_timeout_s)
        await self._emit(
            "state",
            {
                "state": "youtube_openai_plan_result",
                "titleSelectors": len(plan.get("title") or []),
                "descriptionSelectors": len(plan.get("description") or []),
            },
        )

        base_title = [
            "div#textbox[contenteditable='true'][role='textbox'][aria-label*='Add a title' i]",
            "#title-textarea #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-required='true']",
        ]
        base_desc = [
            "div#textbox[contenteditable='true'][role='textbox'][aria-label*='Tell viewers about your video' i]",
            "#description #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-label*='description' i]",
        ]

        selectors = {
            "title": (plan.get("title") or []) + base_title,
            "description": (plan.get("description") or []) + base_desc,
        }

        script = """
(payload) => {
  const normalize = (text) => String(text || '').replace(/\\u200B/g, '').replace(/\\s+/g, ' ').trim();
  const value = String(payload.value || '');
  for (const selector of payload.selectors || []) {
    const el = document.querySelector(selector);
    if (!el) continue;
    try { el.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
    try { el.focus(); } catch (_) {}
    let ok = false;
    try {
      document.execCommand('selectAll', false, null);
      document.execCommand('delete', false, null);
      ok = Boolean(document.execCommand('insertText', false, value));
    } catch (_) {}
    if (!ok) {
      try { el.textContent = value; } catch (_) {}
    }
    try { el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: value, inputType: 'insertText' })); } catch (_) {
      try { el.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
    }
    try { el.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
    const current = normalize(el.innerText || el.textContent || '');
    const expected = normalize(value);
    if (!expected || current === expected || current.includes(expected) || expected.includes(current)) {
      return { ok: true, selector };
    }
  }
  return { ok: false };
}
"""

        out = {"title": not bool(title), "description": not bool(description)}
        if title:
            res = await page.evaluate(script, {"selectors": selectors["title"], "value": title})
            out["title"] = bool((res or {}).get("ok"))
        if description:
            res = await page.evaluate(script, {"selectors": selectors["description"], "value": description})
            out["description"] = bool((res or {}).get("ok"))
        await self._emit("state", {"state": "youtube_openai_fill_result", **out})
        return out

    async def _fill_youtube_fields_via_cdp_replace(self, fields: dict[str, Any]) -> dict[str, bool] | None:
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

        script = """
(data) => {
  const normalize = (text) => String(text || '').replace(/\\u200B/g, '').replace(/\\s+/g, ' ').trim();
  const result = { title: true, description: true };

  const titleEl = document.querySelector("div#textbox[contenteditable='true'][role='textbox'][aria-required='true']")
    || document.querySelector("#title-textarea #textbox[contenteditable='true']");
  const descEl = document.querySelector("div#textbox[contenteditable='true'][role='textbox'][aria-label*='Tell viewers about your video' i]")
    || document.querySelector("#description #textbox[contenteditable='true']");

  const setField = (el, value) => {
    if (!el) return false;
    const text = String(value || '');
    try { el.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
    try { el.focus(); } catch (_) {}
    let applied = false;
    try {
      const sel = window.getSelection?.();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      sel?.removeAllRanges();
      sel?.addRange(range);
      document.execCommand('selectAll', false, null);
      document.execCommand('delete', false, null);
      applied = Boolean(document.execCommand('insertText', false, text));
    } catch (_) {
      applied = false;
    }
    if (!applied) {
      try { el.textContent = text; } catch (_) {}
    }
    try { el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, composed: true, data: text, inputType: 'insertText' })); } catch (_) {}
    try { el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: text, inputType: 'insertText' })); } catch (_) {
      try { el.dispatchEvent(new Event('input', { bubbles: true })); } catch (_) {}
    }
    try { el.dispatchEvent(new Event('change', { bubbles: true })); } catch (_) {}
    try { el.dispatchEvent(new Event('blur', { bubbles: true })); } catch (_) {}

    const expected = normalize(text);
    const current = normalize(el.innerText || el.textContent || '');
    if (!expected) return current.length === 0;
    return current === expected || current.includes(expected) || expected.includes(current);
  };

  if (String(data.title || '').length > 0) {
    const current = normalize(titleEl?.innerText || titleEl?.textContent || '');
    const incoming = normalize(data.title);
    // Replace current auto-filled filename title with requested title.
    if (titleEl && current !== incoming) {
      result.title = setField(titleEl, data.title);
    } else {
      result.title = Boolean(titleEl);
    }
  }

  if (String(data.description || '').length > 0) {
    // Explicitly target YouTube description textbox (aria-label starts with Tell viewers...).
    result.description = setField(descEl, data.description);
  }

  return result;
}
"""
        try:
            payload = await asyncio.wait_for(page.evaluate(script, {"title": title, "description": description}), timeout=8.0)
        except Exception:
            return {"title": False if title else True, "description": False if description else True}

        if not isinstance(payload, dict):
            return {"title": False if title else True, "description": False if description else True}
        return {"title": bool(payload.get("title", not bool(title))), "description": bool(payload.get("description", not bool(description)))}

    async def _youtube_publish_steps_via_cdp(self) -> dict[str, Any]:
        if self.cdp is None:
            await self._connect_cdp()
        if self.cdp is None:
            return {"ok": False, "clicked": {"next": 0, "save": 0}, "error": "CDP unavailable"}

        page = await self.cdp.find_page_by_url_contains("studio.youtube.com")
        if page is None:
            page = await self.cdp.get_most_recent_page()
        if page is None:
            return {"ok": False, "clicked": {"next": 0, "save": 0}, "error": "No page"}

        clicked_next = 0
        clicked_save = 0

        async def _click(selector: str) -> bool:
            locator = page.locator(selector).first
            try:
                if await locator.count() == 0:
                    return False
                await locator.scroll_into_view_if_needed(timeout=1500)
            except Exception:
                pass
            try:
                await locator.click(timeout=3000)
                return True
            except Exception:
                try:
                    await locator.click(timeout=3000, force=True)
                    return True
                except Exception:
                    return False

        for _ in range(3):
            if await _click("button[aria-label='Next']") or await _click("button[aria-label*='Next' i]"):
                clicked_next += 1
            try:
                await page.wait_for_timeout(450)
            except Exception:
                pass

        if await _click("button[aria-label='Save']") or await _click("button[aria-label*='Save' i]"):
            clicked_save += 1

        return {"ok": clicked_save > 0 or clicked_next > 0, "clicked": {"next": clicked_next, "save": clicked_save}, "mode": "cdp_publish_steps"}


    async def _get_x_compose_page(self) -> Any:
        if self.cdp is None:
            await self._connect_cdp()
        if self.cdp is None:
            return None

        page = await self.cdp.find_page_by_url_contains("x.com")
        if page is None:
            page = await self.cdp.get_most_recent_page()
        return page

    async def _fill_x_description_via_cdp(self, description: str) -> dict[str, Any]:
        page = await self._get_x_compose_page()
        if page is None:
            return {"description": False, "mode": "cdp_fill_unavailable", "error": "No X page"}

        value = str(description or "")
        editor_root = page.locator("div.DraftEditor-root").first
        span_target = editor_root.locator("span[data-text='true']").first

        target = editor_root
        target_name = "div.DraftEditor-root"
        try:
            if await span_target.count() > 0:
                target = span_target
                target_name = "div.DraftEditor-root span[data-text='true']"
        except Exception:
            target = editor_root

        try:
            await target.scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass

        try:
            await target.click(timeout=4000, force=True)
        except Exception as exc:
            return {
                "description": False,
                "mode": "cdp_type",
                "reason": "target_click_failed",
                "error": str(exc),
                "target": target_name,
            }

        try:
            await page.keyboard.press("ControlOrMeta+A")
            await page.keyboard.press("Backspace")
            if value:
                await page.keyboard.type(value, delay=12)
        except Exception as exc:
            return {
                "description": False,
                "mode": "cdp_type",
                "reason": "keyboard_type_failed",
                "error": str(exc),
                "target": target_name,
            }

        verify_script = """
() => {
  const normalize = (text) => String(text || '').replace(/\\u00a0/g, ' ').replace(/\\s+/g, ' ').trim();
  const root = document.querySelector('div.DraftEditor-root');
  if (!root) return { current: '', found: false };
  return { current: normalize(root.innerText || root.textContent || ''), found: true };
}
"""
        try:
            verify = await asyncio.wait_for(page.evaluate(verify_script), timeout=4.0)
        except Exception:
            verify = {"current": "", "found": False}

        current = str((verify or {}).get("current") or "")
        expected = " ".join(value.replace(" ", " ").split())
        ok = (not expected and not current) or current == expected or current.startswith(expected) or expected.startswith(current)
        return {
            "description": ok,
            "mode": "cdp_type",
            "reason": "ok" if ok else "value_mismatch",
            "target": target_name,
        }

    async def _submit_x_post_via_cdp(self, wait_for_upload: bool = False, timeout_ms: int = 120000) -> dict[str, Any]:
        page = await self._get_x_compose_page()
        if page is None:
            return {"submitted": False, "mode": "cdp_submit_unavailable", "error": "No X page"}

        bounded_timeout_ms = max(1000, int(timeout_ms or 120000))
        started = time.monotonic()

        script = """
() => {
  const candidates = [
    document.querySelector("button[data-testid='tweetButtonInline']"),
    document.querySelector("button[data-testid='tweetButton']"),
    document.querySelector("div[data-testid='tweetButtonInline']"),
  ].filter(Boolean);

  for (const node of candidates) {
    const ariaDisabled = String(node.getAttribute('aria-disabled') || '').toLowerCase();
    const disabled = Boolean(node.disabled) || ariaDisabled === 'true';
    if (disabled) continue;
    try { node.scrollIntoView({ block: 'center', inline: 'center', behavior: 'instant' }); } catch (_) {}
    try { node.focus({ preventScroll: true }); } catch (_) {}
    try { node.click(); return { submitted: true, mode: 'cdp_submit' }; } catch (_) {}
    try {
      node.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, button: 0, buttons: 1 }));
      return { submitted: true, mode: 'cdp_submit' };
    } catch (_) {}
  }

  return { submitted: false, mode: 'cdp_submit', reason: 'submit_not_clickable' };
}
"""
        last_payload: dict[str, Any] = {"submitted": False, "mode": "cdp_submit", "reason": "submit_not_clickable"}
        while (time.monotonic() - started) * 1000 < bounded_timeout_ms:
            try:
                payload = await asyncio.wait_for(page.evaluate(script), timeout=8.0)
            except Exception as exc:
                return {"submitted": False, "mode": "cdp_submit_exception", "error": str(exc)}

            if not isinstance(payload, dict):
                return {"submitted": False, "mode": "cdp_submit", "reason": "invalid_payload"}

            last_payload = payload
            if bool(payload.get("submitted")):
                payload.setdefault("waitedMs", int((time.monotonic() - started) * 1000))
                return payload

            if not wait_for_upload:
                payload.setdefault("waitedMs", int((time.monotonic() - started) * 1000))
                return payload

            await asyncio.sleep(0.35)

        last_payload.setdefault("waitedMs", int((time.monotonic() - started) * 1000))
        last_payload["reason"] = "submit_not_ready_before_timeout"
        return last_payload

    async def _x_post_status_via_cdp(self) -> dict[str, Any]:
        page = await self._get_x_compose_page()
        if page is None:
            return {"posted": True, "mode": "cdp_status_assumed"}

        script = """
() => {
  const inline = document.querySelector("button[data-testid='tweetButtonInline']");
  if (!inline) return { posted: true, mode: 'cdp_status_assumed', reason: 'composer_closed' };
  const ariaDisabled = String(inline.getAttribute('aria-disabled') || '').toLowerCase();
  const disabled = Boolean(inline.disabled) || ariaDisabled === 'true';
  return { posted: true, mode: 'cdp_status', submitDisabled: disabled };
}
"""
        try:
            payload = await asyncio.wait_for(page.evaluate(script), timeout=6.0)
        except Exception:
            return {"posted": True, "mode": "cdp_status_assumed"}
        if not isinstance(payload, dict):
            return {"posted": True, "mode": "cdp_status_assumed"}
        return payload

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
                page = await self.cdp.find_page_by_url_contains("studio.youtube.com") if platform.lower() == "youtube" else await self.cdp.get_most_recent_page()
                if page is None:
                    page = await self.cdp.get_or_create_page(
                        PLATFORM_URLS.get(platform.lower(), "https://example.com"),
                        reuse_tab=True,
                    )

                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                input_locators = [
                    page.locator("input[type='file']"),
                    page.locator("ytcp-uploads-dialog input[type='file']"),
                    page.locator("input[type='file'][accept*='video']"),
                ]

                mode = None
                last_err = ""
                for _ in range(20):
                    for locator in input_locators:
                        try:
                            count = await locator.count()
                        except Exception:
                            count = 0
                        if count <= 0:
                            continue
                        for idx in range(count):
                            target = locator.nth(idx)
                            try:
                                await target.set_input_files(file_path)
                                mode = "cdp_set_input_files"
                                break
                            except Exception as exc:
                                last_err = str(exc)
                                continue
                        if mode:
                            break
                    if mode:
                        break
                    try:
                        await page.wait_for_timeout(300)
                    except Exception:
                        pass

                if not mode:
                    used_dom_fallback = False
                    if self.cdp is not None:
                        for selector in ["ytcp-uploads-dialog input[type='file']", "input[type='file'][accept*='video']", "input[type='file']"]:
                            for _ in range(4):
                                used_dom_fallback = await self.cdp.set_file_input_files_via_dom(page, selector, file_path)
                                if used_dom_fallback:
                                    mode = "cdp_dom_set_file_input_files"
                                    break
                                try:
                                    await page.wait_for_timeout(250)
                                except Exception:
                                    pass
                            if used_dom_fallback:
                                break

                if not mode:
                    # CDP file-input access is brittle across upload UIs; always try extension debugger fallback.
                    extension_err = ""
                    try:
                        ack = await self._send_extension_cmd("upload.select_file", payload)
                        ack_payload = _ack_from_extension(ack)
                        if bool(ack_payload.get("ok")):
                            mode = str((ack_payload.get("payload") or {}).get("mode") or "extension_debugger_set_file_input_files")
                            await self._emit("state", {"state": "upload_selected", "platform": platform, "filePath": file_path, "mode": mode})
                            return {"ok": True, "payload": {"mode": mode, "fileSizeMb": round(file_size_mb, 2)}}
                        extension_err = str(ack_payload.get("error") or "")
                    except Exception as exc:
                        extension_err = str(exc)

                    reason_parts = [last_err or "file input not found"]
                    if extension_err:
                        reason_parts.append(f"extension fallback failed: {extension_err}")
                    reason = "; ".join(part for part in reason_parts if part)
                    raise RuntimeError(
                        "Automatic upload over remote CDP failed to set the file input"
                        f" (sizeMb={round(file_size_mb, 2)}): {reason}"
                    )

                await self._emit("state", {"state": "upload_selected", "platform": platform, "filePath": file_path, "mode": mode})
                return {"ok": True, "payload": {"mode": mode, "fileSizeMb": round(file_size_mb, 2)}}

            if name == "platform.ensure_logged_in":
                platform = str(payload.get("platform") or "")
                await self._emit("state", {"state": "login_check_skipped", "platform": platform, "loggedIn": True})
                return {"ok": True, "payload": {"platform": platform, "loggedIn": True, "mode": "service_fastpath"}}

            if name == "form.fill":
                platform = str(payload.get("platform") or "").lower()
                fields = payload.get("fields") or {}
                if platform == "youtube" and isinstance(fields, dict):
                    await self._emit(
                        "state",
                        {
                            "state": "youtube_form_fill_start",
                            "openaiKeyPresent": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                            "openaiModel": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                        },
                    )

                    mode = "youtube_openai_fill"
                    try:
                        dom_result = await asyncio.wait_for(self._fill_youtube_fields_via_openai(fields), timeout=12.0)
                    except Exception:
                        dom_result = None

                    if not isinstance(dom_result, dict):
                        mode = "youtube_cdp_replace_fallback"
                        await self._emit("state", {"state": "youtube_form_fill_fallback", "mode": mode})
                        try:
                            dom_result = await asyncio.wait_for(self._fill_youtube_fields_via_cdp_replace(fields), timeout=12.0)
                        except Exception:
                            dom_result = {"title": False, "description": False}

                    title_requested = bool(str(fields.get("title") or ""))
                    description_requested = bool(str(fields.get("description") or ""))
                    title_ok = (not title_requested) or bool((dom_result or {}).get("title", False))
                    description_ok = (not description_requested) or bool((dom_result or {}).get("description", False))
                    await self._emit(
                        "state",
                        {
                            "state": "youtube_form_fill_done",
                            "mode": mode,
                            "title": title_ok,
                            "description": description_ok,
                        },
                    )
                    if title_ok and description_ok:
                        return {"ok": True, "payload": {**(dom_result or {}), "mode": mode}}
                    return {
                        "ok": False,
                        "error": "YouTube form.fill could not set all requested fields",
                        "payload": {**(dom_result or {}), "mode": mode},
                    }

                if platform == "x" and isinstance(fields, dict):
                    description = str(fields.get("description") or "")
                    try:
                        ack = await self._send_extension_cmd(name, payload)
                        return _ack_from_extension(ack)
                    except Exception as exc:
                        if description and ("no extension client connected" in str(exc).lower() or self._is_connection_closed_error(exc)):
                            cdp_result = await self._fill_x_description_via_cdp(description)
                            ok = bool(cdp_result.get("description"))
                            return {
                                "ok": ok,
                                "error": "X form.fill fallback failed" if not ok else "",
                                "payload": cdp_result,
                            }
                        raise

                ack = await self._send_extension_cmd(name, payload)
                return _ack_from_extension(ack)

            if name == "youtube.publish_steps":
                result = await self._youtube_publish_steps_via_cdp()
                return {"ok": bool(result.get("ok", False)), "payload": result}

            if name in {"post.submit", "post.status", "dom.query", "dom.click", "dom.type"}:
                platform = str(payload.get("platform") or "").lower()
                if platform == "x" and name in {"post.submit", "post.status"}:
                    try:
                        ack = await self._send_extension_cmd(name, payload)
                        return _ack_from_extension(ack)
                    except Exception as exc:
                        if "no extension client connected" not in str(exc).lower() and not self._is_connection_closed_error(exc):
                            raise
                        if name == "post.submit":
                            requested_timeout_ms = int(payload.get("timeoutMs") or 0)
                            fallback_timeout_ms = max(120000, requested_timeout_ms or 120000)
                            submit_result = await self._submit_x_post_via_cdp(
                                wait_for_upload=bool(payload.get("waitForUpload", True)),
                                timeout_ms=fallback_timeout_ms,
                            )
                            return {
                                "ok": bool(submit_result.get("submitted")),
                                "error": "X post.submit fallback failed" if not bool(submit_result.get("submitted")) else "",
                                "payload": submit_result,
                            }
                        status_result = await self._x_post_status_via_cdp()
                        return {"ok": True, "payload": status_result}

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
