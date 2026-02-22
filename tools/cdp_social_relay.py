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
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


RELAY_CDP_STEP_TIMEOUT_SECONDS = max(1.0, float(os.getenv("GROK_CDP_RELAY_STEP_TIMEOUT_SECONDS", "6")))
RELAY_LOGS_DIR = Path(os.getenv("GROK_CDP_RELAY_LOG_DIR", "logs/cdp-relay")).expanduser()

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
                with page.expect_file_chooser(timeout=5000) as chooser_info:
                    node.click(timeout=3000)
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
            for timeout_ms in (15000, 30000):
                try:
                    node.set_input_files(video_path, timeout=timeout_ms)
                    return True, f"staged via selector '{selector}' (timeout={timeout_ms}ms)"
                except Exception:
                    try:
                        page.wait_for_timeout(250)
                    except Exception:
                        pass
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
        const isVisible = (node) => {
            if (!node) return false;
            try {
                const style = window.getComputedStyle(node);
                if (!style || style.display === "none" || style.visibility === "hidden") return false;
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            } catch (_) {
                return true;
            }
        };
        const byVisibleSelectors = (selectors) => {
            for (const selector of selectors) {
                try {
                    const nodes = Array.from(document.querySelectorAll(selector));
                    const visible = nodes.find((node) => isVisible(node));
                    if (visible) return visible;
                    if (nodes[0]) return nodes[0];
                } catch (_) {}
            }
            return null;
        };
        const clickNodeOrAncestor = (el) => {
            if (!el) return false;
            const target = el.closest('button, [role="button"]') || el;
            try { target.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
            try {
                target.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true, composed: true, view: window }));
                target.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, composed: true, view: window }));
                target.dispatchEvent(new MouseEvent('pointerup', { bubbles: true, cancelable: true, composed: true, view: window }));
                target.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, composed: true, view: window }));
                target.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, view: window }));
                target.click?.();
                return true;
            } catch (_) {
                try { target.click(); return true; } catch (_) { return false; }
            }
        };

        const captionTarget = byVisibleSelectors([
            '[contenteditable="true"][data-e2e*="caption"]',
            '[contenteditable="true"][aria-label*="caption" i]',
            '[contenteditable="true"][placeholder*="describe" i]',
            '[contenteditable="true"]',
        ]);
        const captionFilled = setText(captionTarget, captionText || titleText);

        const postButton = byVisibleSelectors([
            'button[data-e2e*="post"]',
            'button[aria-label*="post" i]',
            'div[role="button"][aria-label*="post" i]',
            'button[class*="post" i]',
        ]);
        const postByText = Array.from(document.querySelectorAll('button, [role="button"], div'))
            .find((node) => isVisible(node) && String(node.textContent || '').trim().toLowerCase() === 'post');
        const finalPostButton = postButton || postByText || null;

        const postDisabled = Boolean(
            finalPostButton
            && (
                finalPostButton.disabled
                || String(finalPostButton.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
                || String(finalPostButton.getAttribute('disabled') || '').toLowerCase() === 'true'
            )
        );

        const submitClicked = Boolean(finalPostButton && !postDisabled && clickNodeOrAncestor(finalPostButton));
        const draftButton = byVisibleSelectors([
            'button[data-e2e="save_draft_button"]',
            'button[data-e2e*="draft"]',
            'button[aria-label*="draft" i]',
        ]);
        const draftClicked = !submitClicked ? clickNodeOrAncestor(draftButton) : false;
        const url = String(location.href || "");

        return {
            platform: "tiktok",
            captionFilled,
            postFound: Boolean(finalPostButton),
            postEnabled: Boolean(finalPostButton && !postDisabled),
            submitClicked,
            draftClicked,
            done: Boolean(submitClicked || draftClicked || url.includes("tab=draft")),
            status: submitClicked
                ? "TikTok: clicked Post via CDP."
                : (draftClicked
                    ? "TikTok: clicked Save draft via CDP."
                    : `TikTok: form step executed (captionFilled=${Boolean(captionFilled)}, postFound=${Boolean(finalPostButton)}, postEnabled=${Boolean(finalPostButton && !postDisabled)}).`),
        };
        ''')

    if platform == "youtube":
        return (
            common
            + r'''
        const isVisible = (node) => {
            if (!node) return false;
            try {
                const style = window.getComputedStyle(node);
                if (!style || style.display === "none" || style.visibility === "hidden") return false;
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            } catch (_) {
                return true;
            }
        };
        const byVisibleSelectors = (selectors) => {
            for (const selector of selectors) {
                try {
                    const nodes = Array.from(document.querySelectorAll(selector));
                    const visible = nodes.find((node) => isVisible(node));
                    if (visible) return visible;
                    if (nodes[0]) return nodes[0];
                } catch (_) {}
            }
            return null;
        };
        const normText = (value) => String(value || '').replace(/\u200B/g, '').replace(/\s+/g, ' ').trim();
        const findYoutubeTextbox = (kinds, scopedSelectors = []) => {
            for (const scopedSelector of scopedSelectors) {
                const scopedNode = byVisibleSelectors([scopedSelector]);
                if (scopedNode) return scopedNode;
            }
            const nodes = Array.from(document.querySelectorAll('div#textbox[contenteditable="true"], textarea#textbox, [contenteditable="true"][id="textbox"]'));
            for (const node of nodes) {
                const aria = String(node.getAttribute('aria-label') || '').toLowerCase();
                if (kinds.some((kind) => aria.includes(kind)) && isVisible(node)) return node;
            }
            for (const node of nodes) {
                const aria = String(node.getAttribute('aria-label') || '').toLowerCase();
                if (kinds.some((kind) => aria.includes(kind))) return node;
            }
            return null;
        };
        const setYouTubeText = (el, value) => {
            if (!el || !value) return false;
            const nextText = String(value || '').trim();
            if (!nextText) return false;
            try {
                if (el.isContentEditable) {
                    el.focus();
                    try { document.execCommand('selectAll', false, null); } catch (_) {}
                    try { document.execCommand('insertText', false, nextText); } catch (_) {}

                    if (normText(el.textContent) !== normText(nextText)) {
                        try {
                            const selection = window.getSelection();
                            const range = document.createRange();
                            range.selectNodeContents(el);
                            range.deleteContents();
                            const textNode = document.createTextNode(nextText);
                            el.appendChild(textNode);
                            if (selection) {
                                selection.removeAllRanges();
                                const caret = document.createRange();
                                caret.setStart(textNode, textNode.length);
                                caret.collapse(true);
                                selection.addRange(caret);
                            }
                        } catch (_) {
                            try { el.textContent = nextText; } catch (_) {}
                        }
                    }

                    try { el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, composed: true, data: nextText, inputType: 'insertText' })); } catch (_) {}
                    try { el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: nextText, inputType: 'insertText' })); } catch (_) {
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'Enter' }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new Event('blur', { bubbles: true }));
                    return normText(el.textContent) === normText(nextText);
                }
                if ('value' in el) {
                    el.value = nextText;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return normText(el.value) === normText(nextText);
                }
            } catch (_) {}
            return false;
        };

        const activateContainer = (node) => {
            if (!node) return false;
            try { node.scrollIntoView({ block: 'center', inline: 'center' }); } catch (_) {}
            try {
                node.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, composed: true, view: window }));
                node.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, composed: true, view: window }));
                node.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, view: window }));
                node.click?.();
            } catch (_) {}
            return true;
        };
        const findYoutubeTextboxByOuterLabel = (hints) => {
            const containers = Array.from(document.querySelectorAll('ytcp-form-input-container'));
            for (const container of containers) {
                const label = String(container.querySelector('#label-text')?.textContent || '').toLowerCase();
                if (!hints.some((hint) => label.includes(hint))) continue;
                const outer = container.querySelector('#outer, #child-input, #container-content');
                if (outer) activateContainer(outer);
                const textbox = container.querySelector('#textbox[contenteditable="true"], textarea#textbox');
                if (textbox && isVisible(textbox)) return textbox;
                if (textbox) return textbox;
            }
            return null;
        };

        const titleTarget = findYoutubeTextboxByOuterLabel(['title']) || findYoutubeTextbox(['title', 'describes your video'], [
            '#title-textarea #textbox[contenteditable="true"]',
            '[aria-label*="add a title" i]#textbox[contenteditable="true"]',
        ]) || byVisibleSelectors([
            'div#textbox[contenteditable="true"][aria-label*="title" i]',
            'textarea#textbox[aria-label*="title" i]',
        ]);
        const descTarget = findYoutubeTextboxByOuterLabel(['description']) || findYoutubeTextbox(['description', 'tell viewers about your video'], [
            '#description #textbox[contenteditable="true"]',
            '[aria-label*="tell viewers about your video" i]#textbox[contenteditable="true"]',
            '[aria-label*="tell viewers" i][contenteditable="true"]#textbox',
        ]) || byVisibleSelectors([
            '#description div#textbox[contenteditable="true"]',
            'div#textbox[contenteditable="true"][aria-label*="description" i]',
            'div#textbox[contenteditable="true"][aria-label*="tell viewers" i]',
            'textarea#textbox[aria-label*="description" i]',
        ]);

        const titleFilled = setYouTubeText(titleTarget, titleText);
        const captionFilled = setYouTubeText(descTarget, captionText);

        const canSubmit = Boolean((!titleText || titleFilled) && (!captionText || captionFilled));
        const publishButton = byVisibleSelectors([
            'button[aria-label*="publish" i]',
            'button[aria-label*="save" i]',
            'ytcp-button[aria-label*="publish" i] button',
        ]);
        const submitClicked = canSubmit ? click(publishButton) : false;

        return {
            platform: "youtube",
            titleFilled,
            captionFilled,
            canSubmit,
            submitClicked,
            done: Boolean(submitClicked),
            status: submitClicked
                ? "YouTube: clicked publish/save button via CDP."
                : `YouTube: metadata step executed (titleFilled=${Boolean(titleFilled)}, captionFilled=${Boolean(captionFilled)}, canSubmit=${Boolean(canSubmit)}).`,
        };
        ''')

    if platform == "facebook":
        return (
            common
            + r'''
        const isVisible = (node) => {
            if (!node) return false;
            try {
                const style = window.getComputedStyle(node);
                if (!style || style.display === "none" || style.visibility === "hidden") return false;
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            } catch (_) {
                return true;
            }
        };
        const byVisibleSelectors = (selectors) => {
            for (const selector of selectors) {
                try {
                    const nodes = Array.from(document.querySelectorAll(selector));
                    const visible = nodes.find((node) => isVisible(node));
                    if (visible) return visible;
                    if (nodes[0]) return nodes[0];
                } catch (_) {}
            }
            return null;
        };
        const findFacebookNextButton = () => {
            const direct = byVisibleSelectors([
                'div[role="button"][aria-label="Next"]',
                'div[role="button"][aria-label*="next" i]',
                'button[aria-label="Next"]',
                'button[aria-label*="next" i]',
            ]);
            if (direct) return direct;

            const textNodes = Array.from(document.querySelectorAll('span, div, button')).filter((node) => {
                const text = String(node.textContent || "").trim().toLowerCase();
                return text === "next" && isVisible(node);
            });
            for (const textNode of textNodes) {
                const clickable = textNode.closest('div[role="button"], button');
                if (clickable && isVisible(clickable)) return clickable;
            }
            return null;
        };

        const normalizeText = (value) => String(value || '').replace(/\u200B/g, '').replace(/\s+/g, ' ').trim();
        const setFacebookCaption = (el, value) => {
            if (!el || !value) return false;
            const nextText = String(value || "").trim();
            if (!nextText) return false;
            try {
                el.scrollIntoView({ block: 'center', inline: 'center' });
            } catch (_) {}
            try {
                el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, composed: true, view: window }));
                el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, composed: true, view: window }));
                el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, view: window }));
            } catch (_) {}
            try { el.focus(); } catch (_) {}
            try {
                if (el.isContentEditable) {
                    try {
                        const sel = window.getSelection();
                        const range = document.createRange();
                        range.selectNodeContents(el);
                        range.collapse(false);
                        if (sel) {
                            sel.removeAllRanges();
                            sel.addRange(range);
                        }
                    } catch (_) {}

                    try { document.execCommand('selectAll', false, null); } catch (_) {}
                    try { document.execCommand('insertText', false, nextText); } catch (_) {}

                    if (normalizeText(el.textContent) !== normalizeText(nextText)) {
                        try {
                            el.innerHTML = '';
                            const p = document.createElement('p');
                            p.setAttribute('dir', 'auto');
                            const textNode = document.createTextNode(nextText);
                            p.appendChild(textNode);
                            el.appendChild(p);
                            const sel = window.getSelection();
                            if (sel) {
                                sel.removeAllRanges();
                                const caret = document.createRange();
                                caret.setStart(textNode, textNode.length);
                                caret.collapse(true);
                                sel.addRange(caret);
                            }
                        } catch (_) {
                            try { el.textContent = nextText; } catch (_) {}
                        }
                    }

                    try {
                        el.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, composed: true, data: nextText, inputType: 'insertText' }));
                    } catch (_) {}
                    try {
                        el.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: nextText, inputType: 'insertText' }));
                    } catch (_) {
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    el.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a' }));
                    el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a' }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new Event('blur', { bubbles: true }));
                    return normalizeText(el.textContent) === normalizeText(nextText);
                }
                if ('value' in el) {
                    el.value = nextText;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    return normalizeText(el.value) === normalizeText(nextText);
                }
            } catch (_) {}
            return false;
        };

        const nextButton = findFacebookNextButton();
        const captionTarget = byVisibleSelectors([
            '[contenteditable="true"][aria-placeholder*="describe your reel" i]',
            '[contenteditable="true"][aria-label*="describe your reel" i]',
            '[contenteditable="true"][data-lexical-editor="true"]',
            '[contenteditable="true"][aria-label*="write" i]',
            '[contenteditable="true"][role="textbox"]',
        ]);
        const desiredCaption = captionText || titleText;
        const captionFilled = !nextButton ? setFacebookCaption(captionTarget, desiredCaption) : false;

        const postButton = byVisibleSelectors([
            'div[role="button"][aria-label*="post" i]',
            'button[aria-label*="post" i]',
            'div[role="button"][aria-label*="share" i]',
            'button[aria-label*="share" i]',
        ]);

        let nextClicked = false;
        if (nextButton) {
            nextClicked = click(nextButton);
        }

        const captionRequired = Boolean(desiredCaption);
        const canSubmit = !nextButton && (!captionRequired || captionFilled);
        const submitClicked = canSubmit ? click(postButton) : false;
        return {
            platform: "facebook",
            captionFilled,
            nextClicked,
            submitClicked,
            done: Boolean(submitClicked),
            status: submitClicked
                ? "Facebook: clicked post button via CDP."
                : (nextClicked
                    ? "Facebook: clicked Next via CDP."
                    : `Facebook: waiting for description/post step (captionFilled=${Boolean(captionFilled)}).`),
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


def _install_dialog_guards(page) -> None:
    try:
        if getattr(page, "_grok_dialog_guards_installed", False):
            return
        setattr(page, "_grok_dialog_guards_installed", True)
    except Exception:
        pass

    try:
        page.add_init_script(
            """
            (() => {
                try { window.alert = () => {}; } catch (_) {}
                try { window.confirm = () => true; } catch (_) {}
                try { window.prompt = () => ''; } catch (_) {}
                try {
                    window.addEventListener('beforeunload', (event) => {
                        try {
                            event.preventDefault();
                            event.returnValue = '';
                        } catch (_) {}
                    }, true);
                } catch (_) {}
            })();
            """
        )
    except Exception:
        pass

    # Do not actively accept/dismiss dialog events here. Playwright can auto-race
    # dialog close calls and throw `Page.handleJavaScriptDialog: No dialog is showing`.
    # The init script above neutralizes alert/confirm/prompt and beforeunload to
    # prevent most dialogs from opening in the first place.


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

            _install_dialog_guards(page)

            file_staged, file_stage_detail = _stage_file_upload(
                page,
                platform,
                str(payload.get("video_path") or ""),
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
