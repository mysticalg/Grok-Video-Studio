#!/usr/bin/env python3
"""CDP relay endpoint for Grok Video Studio social uploads.

Runs an HTTP endpoint used by app.py and executes best-effort actions in the
currently open QtWebEngine page via Chromium CDP.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


RELAY_CDP_STEP_TIMEOUT_SECONDS = max(1.0, float(os.getenv("GROK_CDP_RELAY_STEP_TIMEOUT_SECONDS", "6")))


def _parse_debug_port(payload: dict[str, Any]) -> int:
    candidate = str(payload.get("qtwebengine_remote_debugging") or "").strip()
    if not candidate:
        candidate = os.getenv("GROK_QTWEBENGINE_REMOTE_DEBUG_PORT", "").strip()
    try:
        port = int(candidate)
    except Exception:
        port = 9222
    return port if port > 0 else 9222


def _pick_page(browser, current_url: str, platform: str):
    pages = []
    for context in browser.contexts:
        pages.extend(context.pages)

    current_url = (current_url or "").strip().lower()
    platform_hints = {
        "tiktok": ["tiktok.com/upload", "tiktokstudio"],
        "youtube": ["studio.youtube.com", "youtube.com/upload", "youtube.com"],
        "facebook": ["facebook.com"],
        "instagram": ["instagram.com"],
    }.get(platform, [])

    if current_url:
        for page in pages:
            url = (page.url or "").lower()
            if current_url in url or url in current_url:
                return page

    for page in pages:
        url = (page.url or "").lower()
        if any(h in url for h in platform_hints):
            return page

    return pages[0] if pages else None



def _upload_selectors_for_platform(platform: str) -> list[str]:
    if platform == "tiktok":
        return [
            'input[type="file"]',
            'input[type="file"][accept*="video"]',
        ]
    if platform == "youtube":
        return [
            'input[type="file"][name="Filedata"]',
            'input[type="file"]',
        ]
    if platform == "facebook":
        return [
            'input[type="file"][accept*="video"]',
            'input[type="file"]',
        ]
    if platform == "instagram":
        return [
            'input[type="file"][accept*="video"]',
            'input[type="file"]',
        ]
    return ['input[type="file"]']


def _upload_trigger_selectors_for_platform(platform: str) -> list[str]:
    if platform == "tiktok":
        return [
            '[data-e2e*="upload"]',
            'button[data-e2e*="upload"]',
            'div[data-e2e*="upload"]',
            'button:has-text("Select video")',
            'button:has-text("Upload")',
            '[role="button"]:has-text("Select file")',
        ]
    if platform == "youtube":
        return [
            'input[type="file"][name="Filedata"] + *',
            'button[aria-label*="select files" i]',
            'ytcp-button[id*="upload"] button',
            'ytcp-button button:has-text("Select files")',
        ]
    if platform == "facebook":
        return [
            'div[role="button"]:has-text("Add video")',
            'button:has-text("Add video")',
            'button:has-text("Upload")',
        ]
    if platform == "instagram":
        return [
            'button:has-text("Select from computer")',
            'button:has-text("Select")',
            'div[role="button"]:has-text("Select")',
        ]
    return ['button:has-text("Upload")', '[role="button"]:has-text("Upload")']


def _prime_upload_surface(page, platform: str) -> str:
    selectors = _upload_trigger_selectors_for_platform(platform)
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = locator.count()
        except Exception:
            continue
        for idx in range(min(count, 3)):
            try:
                locator.nth(idx).click(timeout=700)
                try:
                    page.wait_for_timeout(250)
                except Exception:
                    pass
                return f"clicked upload trigger '{selector}'"
            except Exception:
                continue
    return "no upload trigger clicked"


def _stage_file_upload_via_file_chooser(page, platform: str, video_path: str) -> tuple[bool, str]:
    selectors = _upload_trigger_selectors_for_platform(platform)
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for idx in range(min(count, 4)):
            node = locator.nth(idx)
            try:
                with page.expect_file_chooser(timeout=1200) as chooser_info:
                    node.click(timeout=1000)
                chooser = chooser_info.value
                chooser.set_files(video_path)
                return True, f"staged via file chooser trigger '{selector}'"
            except Exception:
                continue
    return False, "no file chooser trigger matched"


def _simulate_drag_drop_affordance(
    page,
    platform: str,
    *,
    video_base64: str,
    video_name: str,
    video_mime: str,
) -> str:
    if not video_base64:
        return "drag/drop skipped: no video_base64 payload"

    if platform == "tiktok":
        selectors = [
            '[data-e2e*="upload"]',
            '[class*="upload"]',
            '[class*="drop"]',
            'main',
        ]
    else:
        selectors = ['[class*="drop"]', '[data-testid*="upload"]', 'main']

    script = """
        ({selectors, fileB64, fileName, fileMime}) => {
            const isVisible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                const rect = el.getBoundingClientRect();
                return rect.width > 4 && rect.height > 4;
            };
            const b64ToUint8 = (b64) => {
                const bin = atob(b64);
                const arr = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
                return arr;
            };
            const fileData = b64ToUint8(String(fileB64 || ""));
            const file = new File([fileData], String(fileName || 'upload.mp4'), { type: String(fileMime || 'video/mp4') });
            const dt = new DataTransfer();
            dt.items.add(file);

            for (const selector of selectors) {
                const nodes = Array.from(document.querySelectorAll(selector));
                for (const node of nodes) {
                    if (!isVisible(node)) continue;
                    const evts = ['dragenter', 'dragover', 'drop'];
                    for (const evt of evts) {
                        try {
                            node.dispatchEvent(new DragEvent(evt, { bubbles: true, cancelable: true, dataTransfer: dt }));
                        } catch (_) {
                            node.dispatchEvent(new Event(evt, { bubbles: true, cancelable: true }));
                        }
                    }
                    return `simulated file drag/drop on ${selector}`;
                }
            }
            return 'no visible drop target found';
        }
    """
    try:
        result = page.evaluate(
            script,
            {
                "selectors": selectors,
                "fileB64": video_base64,
                "fileName": video_name or "upload.mp4",
                "fileMime": video_mime or "video/mp4",
            },
        )
        return str(result or "drag/drop simulation attempted")
    except Exception as exc:
        return f"drag/drop simulation failed: {exc}"


def _stage_file_upload(
    page,
    platform: str,
    video_path: str,
    *,
    video_base64: str,
    video_name: str,
    video_mime: str,
) -> tuple[bool, str]:
    if not video_path:
        return False, "video_path missing"
    if not os.path.isfile(video_path):
        return False, f"video file not found: {video_path}"

    prime_detail = _prime_upload_surface(page, platform)

    selectors = _upload_selectors_for_platform(platform)
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        for idx in range(count):
            node = locator.nth(idx)
            try:
                node.set_input_files(video_path, timeout=2000)
                return True, f"staged via selector '{selector}'"
            except Exception:
                continue

    chooser_staged, chooser_detail = _stage_file_upload_via_file_chooser(page, platform, video_path)
    if chooser_staged:
        return True, chooser_detail

    dnd_detail = _simulate_drag_drop_affordance(
        page,
        platform,
        video_base64=video_base64,
        video_name=video_name,
        video_mime=video_mime,
    )
    return False, f"no writable file input found; {prime_detail}; {chooser_detail}; {dnd_detail}"

def _script_for_platform(platform: str) -> str:
    common = r'''
        const titleText = String(payload.title || "").trim();
        const captionText = String(payload.caption || "").trim();
        const click = (el) => {
            if (!el) return false;
            try { el.scrollIntoView({block: "center", inline: "center"}); } catch (_) {}
            try { el.click(); return true; } catch (_) { return false; }
        };
        const setText = (el, value) => {
            if (!el || !value) return false;
            try {
                if (el.isContentEditable) {
                    el.focus();
                    document.execCommand("selectAll", false, null);
                    document.execCommand("insertText", false, value);
                    el.dispatchEvent(new Event("input", {bubbles: true}));
                    return true;
                }
                if ("value" in el) {
                    el.value = value;
                    el.dispatchEvent(new Event("input", {bubbles: true}));
                    el.dispatchEvent(new Event("change", {bubbles: true}));
                    return true;
                }
            } catch (_) {}
            return false;
        };
        const bySelectors = (selectors) => {
            for (const selector of selectors) {
                try {
                    const node = document.querySelector(selector);
                    if (node) return node;
                } catch (_) {}
            }
            return null;
        };
    '''

    if platform == "tiktok":
        return (
            common
            + r'''
        const captionTarget = bySelectors([
            '[contenteditable="true"][data-e2e*="caption"]',
            '[contenteditable="true"][aria-label*="caption" i]',
            '[contenteditable="true"]',
        ]);
        const captionFilled = setText(captionTarget, captionText || titleText);

        const postButton = bySelectors([
            'button[data-e2e="save_draft_button"]',
            'button[data-e2e*="post"]',
            'button[aria-label*="post" i]',
            'button:has-text("Post")',
        ]);
        const submitClicked = click(postButton);
        const url = String(location.href || "");

        return {
            platform: "tiktok",
            captionFilled,
            submitClicked,
            done: Boolean(submitClicked || url.includes("tab=draft")),
            status: submitClicked ? "TikTok: clicked post/draft button via CDP." : "TikTok: caption step executed.",
        };
        ''')

    if platform == "youtube":
        return (
            common
            + r'''
        const titleTarget = bySelectors([
            'textarea#textbox[aria-label*="title" i]',
            'div#textbox[contenteditable="true"][aria-label*="title" i]',
            '#textbox[contenteditable="true"]',
        ]);
        const descTarget = bySelectors([
            'textarea#textbox[aria-label*="description" i]',
            'div#textbox[contenteditable="true"][aria-label*="description" i]',
        ]);
        const titleFilled = setText(titleTarget, titleText);
        const captionFilled = setText(descTarget, captionText);

        const publishButton = bySelectors([
            'button[aria-label*="publish" i]',
            'button[aria-label*="save" i]',
            'ytcp-button[aria-label*="publish" i] button',
        ]);
        const submitClicked = click(publishButton);

        return {
            platform: "youtube",
            titleFilled,
            captionFilled,
            submitClicked,
            done: Boolean(submitClicked),
            status: submitClicked ? "YouTube: clicked publish/save button via CDP." : "YouTube: metadata step executed.",
        };
        ''')

    if platform == "facebook":
        return (
            common
            + r'''
        const captionTarget = bySelectors([
            '[contenteditable="true"][aria-label*="write" i]',
            '[contenteditable="true"][role="textbox"]',
        ]);
        const captionFilled = setText(captionTarget, captionText || titleText);
        const postButton = bySelectors([
            'div[role="button"][aria-label*="post" i]',
            'button[aria-label*="post" i]',
        ]);
        const submitClicked = click(postButton);
        return {
            platform: "facebook",
            captionFilled,
            submitClicked,
            done: Boolean(submitClicked),
            status: submitClicked ? "Facebook: clicked post button via CDP." : "Facebook: caption step executed.",
        };
        ''')

    if platform == "instagram":
        return (
            common
            + r'''
        const captionTarget = bySelectors([
            'textarea[aria-label*="caption" i]',
            '[contenteditable="true"][role="textbox"]',
            'textarea',
        ]);
        const captionFilled = setText(captionTarget, captionText || titleText);
        const shareButton = bySelectors([
            'button:has-text("Share")',
            'button[aria-label*="share" i]',
            'button[aria-label*="post" i]',
        ]);
        const submitClicked = click(shareButton);
        return {
            platform: "instagram",
            captionFilled,
            submitClicked,
            done: Boolean(submitClicked),
            status: submitClicked ? "Instagram: clicked share button via CDP." : "Instagram: caption step executed.",
        };
        ''')

    return "return {platform: payload.platform || 'unknown', done: false, status: 'Unsupported platform for CDP relay.'};"


def _handle_with_cdp(payload: dict[str, Any]) -> dict[str, Any]:
    platform = str(payload.get("platform") or "").strip().lower()
    if not platform:
        return {"handled": False, "done": False, "status": "Missing platform."}

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        return {
            "handled": False,
            "done": False,
            "status": "Playwright missing; relay cannot use CDP.",
            "log": f"{exc}",
        }

    port = _parse_debug_port(payload)
    endpoint = f"http://127.0.0.1:{port}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint)
            page = _pick_page(browser, str(payload.get("current_url") or ""), platform)
            if page is None:
                browser.close()
                return {
                    "handled": False,
                    "done": False,
                    "status": f"No CDP page target found for {platform}.",
                    "retry_ms": 1500,
                }

            file_staged, file_stage_detail = _stage_file_upload(
                page,
                platform,
                str(payload.get("video_path") or ""),
                video_base64=str(payload.get("video_base64") or ""),
                video_name=str(payload.get("video_name") or "upload.mp4"),
                video_mime=str(payload.get("video_mime") or "video/mp4"),
            )

            script = _script_for_platform(platform)
            result = page.evaluate(
                """([payload, script]) => {
                    try {
                        return (new Function('payload', script))(payload) || {};
                    } catch (err) {
                        return { done: false, status: 'CDP JS error', error: String(err) };
                    }
                }""",
                [payload, script],
            )
            browser.close()
    except Exception as exc:
        return {
            "handled": False,
            "done": False,
            "status": f"CDP connection/action failed at {endpoint}.",
            "log": str(exc),
            "retry_ms": 1500,
        }

    if not isinstance(result, dict):
        result = {}

    done = bool(result.get("done"))
    status = str(result.get("status") or f"{platform}: CDP step executed")
    if not done and file_staged:
        status = f"{status} File staged via CDP."
    return {
        "handled": True,
        "done": done,
        "status": status,
        "progress": 100 if done else 70,
        "retry_ms": 1200,
        "log": json.dumps({
            "result": result,
            "file_staged": file_staged,
            "file_stage_detail": file_stage_detail,
        }, ensure_ascii=False)[:700],
    }


def _handle_with_cdp_bounded(payload: dict[str, Any]) -> dict[str, Any]:
    result_holder: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result_holder["value"] = _handle_with_cdp(payload)
        except Exception as exc:  # noqa: BLE001
            result_holder["value"] = {
                "handled": False,
                "done": False,
                "status": "CDP relay worker failed.",
                "retry_ms": 1200,
                "log": str(exc),
            }

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=RELAY_CDP_STEP_TIMEOUT_SECONDS)

    if thread.is_alive():
        return {
            "handled": False,
            "done": False,
            "status": (
                "CDP relay step exceeded timeout "
                f"({RELAY_CDP_STEP_TIMEOUT_SECONDS:.1f}s); retrying next step."
            ),
            "retry_ms": 1200,
            "log": "cdp-step-timeout",
        }

    value = result_holder.get("value")
    if isinstance(value, dict):
        return value
    return {"handled": False, "done": False, "status": "CDP relay returned no result.", "retry_ms": 1200}



class RelayHandler(BaseHTTPRequestHandler):
    server_version = "GrokCDPRelay/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {self.client_address[0]} - {fmt % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.error) as exc:
            # Client disconnected before reading response; treat as non-fatal.
            self.log_message("client disconnected before response flush: %s", exc)

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
        debug_port = _parse_debug_port(payload)
        print(
            f"relay step: platform={platform} attempt={attempt} "
            f"debug_port={debug_port} url={current_url}"
        )

        response = _handle_with_cdp_bounded(payload)
        print(
            "relay result: "
            f"handled={bool(response.get('handled'))} done={bool(response.get('done'))} "
            f"status={str(response.get('status') or '')[:160]}"
        )
        self._send_json(200, response)


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
