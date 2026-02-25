from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def _best_effort_type(executor: BaseExecutor, platform: str, selectors: list[str], value: str) -> None:
    text = str(value or "")
    if not text.strip():
        return
    for selector in selectors:
        try:
            result = executor.run(
                "dom.type",
                {
                    "platform": platform,
                    "selector": selector,
                    "value": text,
                },
            )
            payload = (result or {}).get("payload") or {}
            if bool(payload.get("typed", False)):
                return
        except Exception:
            continue



def _best_effort_log_note(executor: BaseExecutor, note: str) -> None:
    try:
        logger = getattr(executor, "_log", None)
        if callable(logger):
            logger("form.fill", "warning", note)
    except Exception:
        return


def run(executor: BaseExecutor, video_path: str, title: str, description: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "youtube", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "youtube"})

    # YouTube Studio often requires opening the Create menu before the upload file input exists.
    _best_effort_click(executor, "youtube", "button[aria-label='Create']")
    _best_effort_click(executor, "youtube", "button[aria-label*='Create' i]")
    _best_effort_click(executor, "youtube", "tp-yt-paper-item[test-id='upload']")
    _best_effort_click(executor, "youtube", "tp-yt-paper-item#text-item-0[test-id='upload']")

    executor.run("upload.select_file", {"platform": "youtube", "filePath": video_path})

    # Avoid long-running form.fill path; use direct dom.type attempts per field.
    _best_effort_type(
        executor,
        "youtube",
        [
            "ytcp-form-input-container#container",
            "ytcp-form-input-container#container #textbox[contenteditable='true']",
            "#title-textarea #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-label*='Add a title' i]",
            "div#textbox[contenteditable='true'][aria-required='true'][aria-label*='title' i]",
        ],
        title,
    )
    _best_effort_type(
        executor,
        "youtube",
        [
            "#description #textbox[contenteditable='true']",
            "div#textbox[contenteditable='true'][aria-label*='Tell viewers about your video' i]",
            "div#textbox[contenteditable='true'][aria-label*='description' i]",
        ],
        description,
    )

    # Keep publish progression DOM-based (same style as embedded/browser-driven flows).
    for _ in range(3):
        _best_effort_click(executor, "youtube", "ytcp-button#next-button button[aria-label='Next']")
        _best_effort_click(executor, "youtube", "ytcp-button#next-button button")
        _best_effort_click(executor, "youtube", "button[aria-label='Next']")
        _best_effort_click(executor, "youtube", "button[aria-label*='Next' i]")

    _best_effort_click(executor, "youtube", "button[aria-label='Save']")
    _best_effort_click(executor, "youtube", "button[aria-label*='Save' i]")

    return executor.run("post.status", {"platform": "youtube"})
