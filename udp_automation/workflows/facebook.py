from __future__ import annotations

import os
import time
from typing import Any

from udp_automation.executors import BaseExecutor


FACEBOOK_STEP_DELAY_SECONDS = max(0.0, float(os.getenv("GROK_FACEBOOK_STEP_DELAY_SECONDS", "1.0")))


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def _best_effort_fill_description(executor: BaseExecutor, description: str) -> None:
    selectors = [
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='Describe your reel' i]",
        "[contenteditable='true'][aria-placeholder*='Describe your reel' i]",
    ]

    value = str(description or "")
    if not value:
        return

    for selector in selectors:
        try:
            executor.run("dom.click", {"platform": "facebook", "selector": selector, "timeoutMs": 8000})
        except Exception:
            continue

        try:
            executor.run(
                "dom.type",
                {
                    "platform": "facebook",
                    "selector": selector,
                    "value": value,
                    "timeoutMs": 8000,
                },
            )
            return
        except Exception:
            continue

    # Final fallback to generic form.fill path.
    try:
        executor.run("form.fill", {"platform": "facebook", "fields": {"description": value}})
    except Exception:
        return


def _sleep_between_actions() -> None:
    if FACEBOOK_STEP_DELAY_SECONDS > 0:
        time.sleep(FACEBOOK_STEP_DELAY_SECONDS)


def run(executor: BaseExecutor, video_path: str, caption: str, title: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "facebook", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "facebook"})

    # Facebook Reels often requires entering the upload surface before file input is present.
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Upload'][aria-label*='reel' i]")
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel']")

    executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})

    _sleep_between_actions()

    # Move through the Facebook reels flow before filling the description.
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Next' i]")
    _sleep_between_actions()
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Next' i]")

    _sleep_between_actions()
    _best_effort_fill_description(executor, caption)

    _sleep_between_actions()
    _best_effort_click(executor, "facebook", "div[aria-label='Post'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Post' i]")

    # Keep standard submit path as a fallback in case the direct click misses.
    executor.run("post.submit", {"platform": "facebook"})
    return executor.run("post.status", {"platform": "facebook"})
