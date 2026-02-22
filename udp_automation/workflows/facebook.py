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


def run(executor: BaseExecutor, video_path: str, caption: str, title: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "facebook", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "facebook"})

    # Facebook Reels often requires entering the upload surface before file input is present.
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Upload'][aria-label*='reel' i]")
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel']")

    executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})

    # Facebook reel flow may require two Next clicks before description input is enabled.
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label='Next']")
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='next' i]")
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label='Next']")

    _best_effort_type(executor, "facebook", "[contenteditable='true'][aria-placeholder*='describe your reel' i]", caption)
    _best_effort_type(executor, "facebook", "[contenteditable='true'][data-lexical-editor='true']", caption)

    executor.run("form.fill", {"platform": "facebook", "fields": {"title": title, "description": caption}})
    executor.run("post.submit", {"platform": "facebook"})
    return executor.run("post.status", {"platform": "facebook"})
