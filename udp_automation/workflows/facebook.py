from __future__ import annotations

import os
import time
from typing import Any

from udp_automation.executors import BaseExecutor


FACEBOOK_STEP_DELAY_SECONDS = max(0.0, float(os.getenv("GROK_FACEBOOK_STEP_DELAY_SECONDS", "1.0")))
FACEBOOK_FORM_FILL_ATTEMPTS = max(1, int(os.getenv("GROK_FACEBOOK_FORM_FILL_ATTEMPTS", "3")))


def _best_effort_log(executor: BaseExecutor, action: str, status: str, note: str) -> None:
    try:
        logger = getattr(executor, "_log", None)
        if callable(logger):
            logger(action, status, note)
    except Exception:
        return


def _best_effort_log_note(executor: BaseExecutor, note: str) -> None:
    _best_effort_log(executor, "facebook.workflow", "info", note)


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def _sleep_between_actions(executor: BaseExecutor, reason: str) -> None:
    if FACEBOOK_STEP_DELAY_SECONDS <= 0:
        return
    _best_effort_log_note(executor, f"facebook workflow delay {FACEBOOK_STEP_DELAY_SECONDS:.1f}s before {reason}")
    time.sleep(FACEBOOK_STEP_DELAY_SECONDS)


def _attempt_fill_description(executor: BaseExecutor, description: str) -> bool:
    selectors = [
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='Describe your reel' i]",
        "[contenteditable='true'][aria-placeholder*='Describe your reel' i]",
    ]

    value = str(description or "").strip()
    if not value:
        _best_effort_log_note(executor, "facebook description is empty; skipping description fill")
        return True

    for attempt in range(1, FACEBOOK_FORM_FILL_ATTEMPTS + 1):
        _best_effort_log_note(
            executor,
            f"facebook form.fill(description) attempt {attempt}/{FACEBOOK_FORM_FILL_ATTEMPTS} len={len(value)}",
        )

        for selector in selectors:
            try:
                executor.run("dom.click", {"platform": "facebook", "selector": selector, "timeoutMs": 8000})
            except Exception:
                continue

        # Primary path: platform-aware form.fill.
        try:
            response = executor.run("form.fill", {"platform": "facebook", "fields": {"description": value}})
            payload = response.get("payload") or {}
            fill_ok = bool(payload.get("description") is True or payload.get("description") == 1)
            if fill_ok:
                _best_effort_log_note(
                    executor,
                    f"facebook form.fill(description) attempt {attempt}/{FACEBOOK_FORM_FILL_ATTEMPTS} succeeded",
                )
                return True
            _best_effort_log(
                executor,
                "form.fill",
                "warning",
                f"facebook form.fill(description) attempt {attempt} returned non-success payload={payload}",
            )
        except Exception as exc:
            _best_effort_log(executor, "form.fill", "warning", f"facebook form.fill(description) attempt {attempt} failed: {exc}")

        # Fallback path: direct typing into known selectors.
        for selector in selectors:
            try:
                type_response = executor.run(
                    "dom.type",
                    {
                        "platform": "facebook",
                        "selector": selector,
                        "value": value,
                        "timeoutMs": 8000,
                    },
                )
                type_payload = type_response.get("payload") or {}
                typed_ok = bool(type_payload.get("typed") is True or type_payload.get("typed") == 1)
                if typed_ok:
                    _best_effort_log_note(executor, f"facebook dom.type fallback succeeded on selector={selector}")
                    return True
            except Exception:
                continue

        if attempt < FACEBOOK_FORM_FILL_ATTEMPTS:
            _sleep_between_actions(executor, f"facebook description fill retry {attempt + 1}")

    return False


def run(executor: BaseExecutor, video_path: str, caption: str, title: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "facebook", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "facebook"})

    # Facebook Reels often requires entering the upload surface before file input is present.
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Upload'][aria-label*='reel' i]")
    _best_effort_click(executor, "facebook", "div[aria-label='Upload video for reel']")

    executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})

    _sleep_between_actions(executor, "facebook reels next step 1")

    # Move through the Facebook reels flow before filling the description.
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Next' i]")
    _sleep_between_actions(executor, "facebook reels next step 2")
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Next' i]")

    _sleep_between_actions(executor, "facebook description fill")
    description_ok = _attempt_fill_description(executor, caption)
    if not description_ok:
        raise RuntimeError(
            "Facebook description fill failed; aborting before submit to avoid posting without caption"
        )

    _sleep_between_actions(executor, "facebook post submit")
    _best_effort_click(executor, "facebook", "div[aria-label='Post'][role='button']")
    _best_effort_click(executor, "facebook", "[role='button'][aria-label*='Post' i]")

    # Keep standard submit path as a fallback in case the direct click misses.
    executor.run("post.submit", {"platform": "facebook"})
    return executor.run("post.status", {"platform": "facebook"})
