from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def _best_effort_type(executor: BaseExecutor, platform: str, selector: str, value: str) -> None:
    if not str(value or "").strip():
        return
    try:
        executor.run("dom.type", {"platform": platform, "selector": selector, "value": value, "timeoutMs": 10000})
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

    # Fallback typing path for YouTube Studio's duplicated contenteditable #textbox fields.
    _best_effort_click(executor, "youtube", "ytcp-form-input-container #outer")
    _best_effort_type(executor, "youtube", "ytcp-form-input-container #label-text + * #textbox[contenteditable='true']", title)
    _best_effort_type(executor, "youtube", "#title-textarea #textbox[contenteditable='true']", title)
    _best_effort_type(executor, "youtube", "#description #textbox[contenteditable='true']", description)
    _best_effort_type(executor, "youtube", "div#textbox[contenteditable='true'][aria-label*='tell viewers' i]", description)

    executor.run("form.fill", {"platform": "youtube", "fields": {"title": title, "description": description}})

    # Advance upload dialog if still on details steps.
    for selector in (
        "ytcp-button#next-button button",
        "ytcp-button[id='next-button'] button",
        "button[aria-label*='next' i]",
    ):
        _best_effort_click(executor, "youtube", selector)

    executor.run("post.submit", {"platform": "youtube"})
    return executor.run("post.status", {"platform": "youtube"})
