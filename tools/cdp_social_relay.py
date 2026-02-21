#!/usr/bin/env python3
"""CDP relay endpoint for Grok Video Studio social uploads.

Runs an HTTP endpoint used by app.py and executes best-effort actions in the
currently open QtWebEngine page via Chromium CDP.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket

import requests
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


RELAY_CDP_STEP_TIMEOUT_SECONDS = max(1.0, float(os.getenv("GROK_CDP_RELAY_STEP_TIMEOUT_SECONDS", "6")))
RELAY_LOGS_DIR = Path(os.getenv("GROK_CDP_RELAY_LOG_DIR", "logs/cdp-relay")).expanduser()
RELAY_NETWORK_REPLAY_ENABLED = os.getenv("GROK_CDP_RELAY_ENABLE_NETWORK_REPLAY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RELAY_AI_ACTIONS_ENABLED = os.getenv("GROK_CDP_RELAY_ENABLE_AI_ACTIONS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RELAY_AI_MODEL = os.getenv("GROK_CDP_RELAY_AI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
RELAY_AI_BASE_URL = os.getenv("GROK_CDP_RELAY_AI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
RELAY_AI_TIMEOUT_SECONDS = max(5.0, float(os.getenv("GROK_CDP_RELAY_AI_TIMEOUT_SECONDS", "20")))

_PLAYWRIGHT_INSTANCE = None
_CDP_BROWSERS_BY_ENDPOINT: dict[str, Any] = {}
_CDP_CONNECT_LOCK = threading.Lock()
_CDP_ACTION_LOCK = threading.Lock()


def _is_context_mgmt_unsupported_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return (
        "browser.setdownloadbehavior" in message
        and "browser context management is not supported" in message
    )


def _close_cdp_runtime() -> None:
    global _PLAYWRIGHT_INSTANCE
    with _CDP_CONNECT_LOCK:
        for endpoint, browser in list(_CDP_BROWSERS_BY_ENDPOINT.items()):
            try:
                browser.close()
            except Exception:
                pass
            _CDP_BROWSERS_BY_ENDPOINT.pop(endpoint, None)
        if _PLAYWRIGHT_INSTANCE is not None:
            try:
                _PLAYWRIGHT_INSTANCE.stop()
            except Exception:
                pass
            _PLAYWRIGHT_INSTANCE = None


atexit.register(_close_cdp_runtime)


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload or {})
    if "caption" in redacted:
        redacted["caption"] = str(redacted.get("caption") or "")[:500]
    if "title" in redacted:
        redacted["title"] = str(redacted.get("title") or "")[:300]
    return redacted


def _append_relay_log(event: str, payload: dict[str, Any]) -> None:
    try:
        RELAY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = RELAY_LOGS_DIR / f"relay-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "payload": payload,
        }
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _wants_network_replay(payload: dict[str, Any]) -> bool:
    candidate = payload.get("use_network_relay_actions")
    if candidate is None:
        return RELAY_NETWORK_REPLAY_ENABLED
    if isinstance(candidate, bool):
        return candidate
    return str(candidate).strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_replay_headers(headers: dict[str, Any]) -> dict[str, str]:
    blocked = {
        "host",
        "content-length",
        "connection",
        "origin",
        "referer",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
    }
    sanitized: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if not key:
            continue
        lower = str(key).strip().lower()
        if lower in blocked:
            continue
        sanitized[str(key)] = str(value)
    return sanitized


def _capture_network_candidates(page, platform: str, capture_ms: int = 1500) -> list[dict[str, Any]]:
    try:
        cdp_session = page.context.new_cdp_session(page)
    except Exception:
        return []

    requests: dict[str, dict[str, Any]] = {}
    captured: list[dict[str, Any]] = []

    platform_hints = {
        "youtube": ("youtubei/v1", "graphql", "upload"),
        "tiktok": ("graphql", "api", "post"),
        "facebook": ("graphql", "api", "composer"),
        "instagram": ("graphql", "api", "media"),
    }.get(platform, ("graphql", "api", "upload"))

    def _on_request(params: dict[str, Any]) -> None:
        request = params.get("request") or {}
        request_id = str(params.get("requestId") or "")
        url = str(request.get("url") or "")
        method = str(request.get("method") or "GET").upper()
        resource_type = str(params.get("type") or "")
        lower_url = url.lower()
        if method not in {"POST", "PUT", "PATCH"}:
            return
        if resource_type not in {"XHR", "Fetch", "Other"}:
            return
        if not any(hint in lower_url for hint in platform_hints):
            return
        requests[request_id] = {
            "request_id": request_id,
            "url": url,
            "method": method,
            "headers": request.get("headers") or {},
            "post_data": request.get("postData") or "",
            "resource_type": resource_type,
        }

    def _on_response(params: dict[str, Any]) -> None:
        request_id = str(params.get("requestId") or "")
        base = requests.get(request_id)
        if not base:
            return
        response = params.get("response") or {}
        mime_type = str(response.get("mimeType") or "")
        if not any(token in mime_type.lower() for token in ("json", "graphql", "javascript", "text")):
            return
        captured.append(
            {
                **base,
                "status": int(response.get("status") or 0),
                "response_mime": mime_type,
            }
        )

    try:
        cdp_session.send("Network.enable")
        cdp_session.on("Network.requestWillBeSent", _on_request)
        cdp_session.on("Network.responseReceived", _on_response)
        page.wait_for_timeout(max(250, capture_ms))
    except Exception:
        return []
    finally:
        try:
            cdp_session.detach()
        except Exception:
            pass

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in captured:
        key = (str(item.get("method") or ""), str(item.get("url") or ""), str(item.get("post_data") or "")[:200])
        deduped[key] = item
    return list(deduped.values())[:8]


def _replay_network_candidates(page, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_results: list[dict[str, Any]] = []
    for candidate in candidates:
        url = str(candidate.get("url") or "")
        method = str(candidate.get("method") or "POST")
        if not url:
            continue
        headers = _sanitize_replay_headers(candidate.get("headers") or {})
        post_data = str(candidate.get("post_data") or "")
        try:
            result = page.evaluate(
                """async ([url, method, headers, body]) => {
                    try {
                        const response = await fetch(url, {
                            method,
                            headers,
                            credentials: 'include',
                            body: body || undefined,
                        });
                        return {
                            ok: response.ok,
                            status: response.status,
                            redirected: response.redirected,
                            url: response.url || url,
                        };
                    } catch (error) {
                        return { ok: false, status: 0, error: String(error), url };
                    }
                }""",
                [url, method, headers, post_data],
            )
        except Exception as exc:
            result = {"ok": False, "status": 0, "error": str(exc), "url": url}

        replay_results.append(
            {
                "url": url,
                "method": method,
                "status": int((result or {}).get("status") or 0),
                "ok": bool((result or {}).get("ok")),
                "error": str((result or {}).get("error") or "")[:200],
            }
        )
        if len(replay_results) >= int(payload.get("max_network_replays") or 3):
            break
    return replay_results


def _wants_ai_actions(payload: dict[str, Any]) -> bool:
    candidate = payload.get("use_ai_relay_actions")
    if candidate is None:
        return RELAY_AI_ACTIONS_ENABLED
    if isinstance(candidate, bool):
        return candidate
    return str(candidate).strip().lower() in {"1", "true", "yes", "on"}


def _relay_ai_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GROK_API_KEY", "").strip()
        or os.getenv("XAI_API_KEY", "").strip()
    )


def _build_ai_dom_snapshot(page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
            const take = (nodes, limit = 40) => Array.from(nodes || []).slice(0, limit);
            const toText = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
            const buttons = take(document.querySelectorAll('button, [role="button"], tp-yt-paper-item')).map((node) => ({
                text: toText(node.textContent).slice(0, 120),
                aria: toText(node.getAttribute('aria-label')).slice(0, 120),
                id: toText(node.id).slice(0, 80),
                className: toText(node.className).slice(0, 120),
                disabled: !!node.disabled,
            }));
            const inputs = take(document.querySelectorAll('input, textarea, [contenteditable="true"]')).map((node) => ({
                tag: String(node.tagName || '').toLowerCase(),
                type: String(node.getAttribute?.('type') || ''),
                name: String(node.getAttribute?.('name') || ''),
                aria: toText(node.getAttribute?.('aria-label')).slice(0, 120),
                placeholder: toText(node.getAttribute?.('placeholder')).slice(0, 120),
                editable: !!node.isContentEditable,
            }));
            return {
                url: String(location.href || ''),
                title: String(document.title || ''),
                buttons,
                inputs,
            };
        }"""
    )


def _request_ai_action_script(payload: dict[str, Any], dom_snapshot: dict[str, Any]) -> dict[str, Any]:
    api_key = _relay_ai_api_key()
    if not api_key:
        return {"ok": False, "error": "AI relay actions enabled, but OPENAI_API_KEY/GROK_API_KEY is not set."}

    instruction = {
        "platform": str(payload.get("platform") or ""),
        "title": str(payload.get("title") or "")[:200],
        "caption": str(payload.get("caption") or "")[:400],
        "goal": "Complete the next safe step in upload/posting flow (fill metadata and click Next/Post/Publish/Share if visible).",
        "rules": [
            "Return strict JSON only.",
            "Provide JS expression body for function(payload){...} without markdown fences.",
            "No loops waiting forever; no navigation to external sites; no credential extraction.",
            "Set done=true only if publish/share/post click was attempted.",
        ],
    }
    messages = [
        {
            "role": "system",
            "content": "You are a browser automation planner. Output JSON: {status:string, done:boolean, script:string}.",
        },
        {
            "role": "user",
            "content": json.dumps({"instruction": instruction, "dom": dom_snapshot}, ensure_ascii=False),
        },
    ]

    body = {
        "model": RELAY_AI_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            f"{RELAY_AI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=RELAY_AI_TIMEOUT_SECONDS,
        )
        data = response.json() if response.content else {}
        if not response.ok:
            return {"ok": False, "error": f"AI request failed HTTP {response.status_code}"}
        content = (
            (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
            if isinstance(data, dict)
            else ""
        )
        parsed = json.loads(content) if content else {}
        script = str(parsed.get("script") or "").strip()
        if not script:
            return {"ok": False, "error": "AI response did not include script."}
        return {
            "ok": True,
            "status": str(parsed.get("status") or "AI action planned."),
            "done": bool(parsed.get("done")),
            "script": script[:6000],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _run_ai_relay_actions(page, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        snapshot = _build_ai_dom_snapshot(page)
    except Exception as exc:
        return {"enabled": True, "ok": False, "error": f"snapshot failed: {exc}"}

    ai_plan = _request_ai_action_script(payload, snapshot)
    if not ai_plan.get("ok"):
        return {"enabled": True, "ok": False, "error": str(ai_plan.get("error") or "unknown ai error")}

    try:
        ai_result = page.evaluate(
            """([payload, script]) => {
                try {
                    const out = (new Function('payload', script))(payload);
                    if (out && typeof out === 'object') return out;
                    return { done: false, status: 'AI script ran without structured output.' };
                } catch (err) {
                    return { done: false, status: 'AI script execution failed', error: String(err) };
                }
            }""",
            [payload, str(ai_plan.get("script") or "")],
        )
    except Exception as exc:
        return {"enabled": True, "ok": False, "error": f"evaluate failed: {exc}"}

    return {
        "enabled": True,
        "ok": True,
        "planned_status": str(ai_plan.get("status") or "AI action planned."),
        "planned_done": bool(ai_plan.get("done")),
        "result": ai_result if isinstance(ai_result, dict) else {},
    }


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
            'button[aria-label*="create" i]',
            'ytcp-button#create-icon button',
            'ytcp-button[id="create-icon"] button',
            'ytcp-button[aria-label*="create" i] button',
            'yt-touch-feedback-shape.yt-spec-touch-feedback-shape--touch-response',
            'tp-yt-paper-item[test-id="upload"]',
            'tp-yt-paper-item#text-item-0[test-id="upload"]',
            'tp-yt-paper-item:has-text("Upload videos")',
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
    if platform == "youtube":
        try:
            create_result = page.evaluate(
                """
                () => {
                    const selectors = [
                        'button[aria-label*="create" i]',
                        'ytcp-button#create-icon button',
                        'ytcp-button[id="create-icon"] button',
                        'ytcp-button[aria-label*="create" i] button',
                        'tp-yt-paper-icon-button[aria-label*="create" i]',
                        'yt-touch-feedback-shape.yt-spec-touch-feedback-shape--touch-response',
                    ];
                    const fireClick = (node) => {
                        if (!node) return false;
                        try { node.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
                        const target = node.closest('button, [role="button"], tp-yt-paper-item, ytcp-button') || node;
                        const events = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'];
                        try {
                            target.focus?.();
                            for (const name of events) {
                                target.dispatchEvent(new MouseEvent(name, { bubbles: true, cancelable: true, composed: true, view: window }));
                            }
                            target.click?.();
                            return true;
                        } catch (_) {
                            try { target.click(); return true; } catch (_) { return false; }
                        }
                    };

                    for (const selector of selectors) {
                        const node = document.querySelector(selector);
                        if (!node) continue;
                        if (fireClick(node)) {
                            return { clicked: true, selector };
                        }
                    }
                    return { clicked: false, selector: '' };
                }
                """
            )
        except Exception:
            create_result = {"clicked": False, "selector": ""}

        create_clicked = bool((create_result or {}).get("clicked"))
        if create_clicked:
            try:
                page.wait_for_timeout(350)
            except Exception:
                pass

        try:
            upload_result = page.evaluate(
                """
                () => {
                    const candidates = Array.from(document.querySelectorAll('tp-yt-paper-item, [role="menuitem"], ytcp-ve'));
                    const fireClick = (node) => {
                        if (!node) return false;
                        try { node.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
                        const target = node.closest('tp-yt-paper-item, [role="menuitem"], button, [role="button"]') || node;
                        const events = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'];
                        try {
                            target.focus?.();
                            for (const name of events) {
                                target.dispatchEvent(new MouseEvent(name, { bubbles: true, cancelable: true, composed: true, view: window }));
                            }
                            target.click?.();
                            return true;
                        } catch (_) {
                            try { target.click(); return true; } catch (_) { return false; }
                        }
                    };

                    const selectorMatches = [
                        'tp-yt-paper-item[test-id="upload"]',
                        'tp-yt-paper-item#text-item-0[test-id="upload"]',
                        '[role="menuitem"][test-id="upload"]',
                    ];
                    for (const selector of selectorMatches) {
                        const node = document.querySelector(selector);
                        if (!node) continue;
                        if (fireClick(node)) return { clicked: true, via: selector };
                    }

                    for (const node of candidates) {
                        const text = String(node.textContent || '').toLowerCase().replace(/\\s+/g, ' ').trim();
                        if (!text.includes('upload videos')) continue;
                        if (fireClick(node)) return { clicked: true, via: 'text:upload videos' };
                    }

                    return { clicked: false, via: '' };
                }
                """
            )
        except Exception:
            upload_result = {"clicked": False, "via": ""}

        if bool((upload_result or {}).get("clicked")):
            try:
                page.wait_for_timeout(350)
            except Exception:
                pass
            if create_clicked:
                return "clicked YouTube create + upload videos menu"
            return "clicked YouTube upload videos menu"

        if create_clicked:
            return "clicked YouTube create trigger"

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


def _stage_file_upload(
    page,
    platform: str,
    video_path: str,
    *,
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

    return False, f"no writable file input found; {prime_detail}; {chooser_detail}"

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


def _get_cdp_browser(endpoint: str):
    global _PLAYWRIGHT_INSTANCE
    from playwright.sync_api import sync_playwright  # type: ignore

    with _CDP_CONNECT_LOCK:
        if _PLAYWRIGHT_INSTANCE is None:
            _PLAYWRIGHT_INSTANCE = sync_playwright().start()

        cached = _CDP_BROWSERS_BY_ENDPOINT.get(endpoint)
        if cached is not None:
            try:
                _ = len(cached.contexts)
                return cached
            except Exception:
                _CDP_BROWSERS_BY_ENDPOINT.pop(endpoint, None)

        browser = _PLAYWRIGHT_INSTANCE.chromium.connect_over_cdp(endpoint)
        _CDP_BROWSERS_BY_ENDPOINT[endpoint] = browser
        return browser


def _handle_with_cdp(payload: dict[str, Any]) -> dict[str, Any]:
    platform = str(payload.get("platform") or "").strip().lower()
    if not platform:
        return {"handled": False, "done": False, "status": "Missing platform."}

    port = _parse_debug_port(payload)
    endpoint = f"http://127.0.0.1:{port}"

    try:
        browser = _get_cdp_browser(endpoint)
        with _CDP_ACTION_LOCK:
            page = _pick_page(browser, str(payload.get("current_url") or ""), platform)
            if page is None:
                return {
                    "handled": False,
                    "done": False,
                    "status": f"No CDP page target found for {platform}.",
                    "retry_ms": 1500,
                }

            try:
                page.set_default_timeout(int(RELAY_CDP_STEP_TIMEOUT_SECONDS * 1000))
            except Exception:
                pass

            file_staged, file_stage_detail = _stage_file_upload(
                page,
                platform,
                str(payload.get("video_path") or ""),
                video_name=str(payload.get("video_name") or "upload.mp4"),
                video_mime=str(payload.get("video_mime") or "video/mp4"),
            )

            network_capture_detail = {
                "enabled": False,
                "captured": 0,
                "replayed": 0,
                "results": [],
            }
            if _wants_network_replay(payload):
                captured_candidates = _capture_network_candidates(
                    page,
                    platform,
                    capture_ms=int(payload.get("network_capture_ms") or 1400),
                )
                replay_results = _replay_network_candidates(page, payload, captured_candidates)
                network_capture_detail = {
                    "enabled": True,
                    "captured": len(captured_candidates),
                    "replayed": len(replay_results),
                    "results": replay_results,
                }

            ai_action_detail = {
                "enabled": False,
                "ok": False,
                "status": "",
                "done": False,
                "error": "",
                "result": {},
            }
            if _wants_ai_actions(payload):
                ai_exec = _run_ai_relay_actions(page, payload)
                ai_result = ai_exec.get("result") if isinstance(ai_exec, dict) else {}
                ai_action_detail = {
                    "enabled": True,
                    "ok": bool(ai_exec.get("ok")) if isinstance(ai_exec, dict) else False,
                    "status": str(
                        (ai_exec.get("planned_status") if isinstance(ai_exec, dict) else "")
                        or (ai_result.get("status") if isinstance(ai_result, dict) else "")
                        or ""
                    ),
                    "done": bool(
                        (ai_result.get("done") if isinstance(ai_result, dict) else False)
                        or (ai_exec.get("planned_done") if isinstance(ai_exec, dict) else False)
                    ),
                    "error": str((ai_exec.get("error") if isinstance(ai_exec, dict) else "") or ""),
                    "result": ai_result if isinstance(ai_result, dict) else {},
                }

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
    except Exception as exc:
        if _is_context_mgmt_unsupported_error(exc):
            return {
                "handled": False,
                "done": False,
                "status": (
                    "CDP unavailable: embedded browser does not support Browser.setDownloadBehavior "
                    "during connect_over_cdp; falling back to non-CDP upload flow."
                ),
                "log": str(exc),
                "retry_ms": 5000,
            }
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
    if ai_action_detail.get("enabled"):
        if ai_action_detail.get("ok"):
            done = done or bool(ai_action_detail.get("done"))
            ai_status = str(ai_action_detail.get("status") or "AI-assisted relay action executed.").strip()
            status = f"{status} {ai_status}".strip()
        else:
            status = f"{status} AI relay unavailable ({str(ai_action_detail.get('error') or 'unknown error')[:120]})."
    if network_capture_detail.get("enabled"):
        status = (
            f"{status} Network replay captured={network_capture_detail.get('captured', 0)} "
            f"replayed={network_capture_detail.get('replayed', 0)}."
        )
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
            "network_replay": network_capture_detail,
            "ai_actions": ai_action_detail,
        }, ensure_ascii=False)[:700],
    }


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
            f"debug_port={debug_port} url={current_url}",
            flush=True,
        )
        _append_relay_log("request", {
            "platform": platform,
            "attempt": attempt,
            "debug_port": debug_port,
            "current_url": current_url,
            "payload": _redact_payload(payload),
        })

        response = _handle_with_cdp(payload)
        print(
            "relay result: "
            f"handled={bool(response.get('handled'))} done={bool(response.get('done'))} "
            f"status={str(response.get('status') or '')[:160]}",
            flush=True,
        )
        _append_relay_log("response", {
            "platform": platform,
            "attempt": attempt,
            "handled": bool(response.get("handled")),
            "done": bool(response.get("done")),
            "status": str(response.get("status") or ""),
            "retry_ms": int(response.get("retry_ms", 0) or 0),
            "progress": int(response.get("progress", 0) or 0),
            "log": str(response.get("log") or "")[:1200],
        })
        self._send_json(200, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CDP relay for social upload steps.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), RelayHandler)
    print(f"CDP relay listening on http://{args.host}:{args.port}/social-upload-step", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
