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


def _best_effort_click(
    executor: BaseExecutor,
    platform: str,
    selector: str,
    *,
    timeout_ms: int = 8000,
    text_contains: str = "",
) -> None:
    payload: dict[str, Any] = {"platform": platform, "selector": selector, "timeoutMs": timeout_ms}
    if text_contains:
        payload["textContains"] = text_contains
    try:
        executor.run("dom.click", payload)
    except Exception:
        return


def _sleep_between_actions(executor: BaseExecutor, reason: str) -> None:
    if FACEBOOK_STEP_DELAY_SECONDS <= 0:
        return
    _best_effort_log_note(executor, f"facebook workflow delay {FACEBOOK_STEP_DELAY_SECONDS:.1f}s before {reason}")
    time.sleep(FACEBOOK_STEP_DELAY_SECONDS)


def _safe_facebook_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if candidate.lower().startswith(("http://", "https://")):
        return candidate
    return "https://www.facebook.com/"




def _wait_for_query_match(
    executor: BaseExecutor,
    platform: str,
    selector: str,
    *,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
) -> bool:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    poll_s = max(0.1, float(poll_s))
    while time.monotonic() < deadline:
        try:
            result = executor.run("dom.query", {"platform": platform, "selector": selector})
            payload = result.get("payload") or {}
            if int(payload.get("count") or 0) > 0:
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def _wait_for_facebook_upload_ready(executor: BaseExecutor) -> None:
    # Wait until progress reaches 100% (or upload indicator clears).
    try:
        executor.run(
            "dom.click",
            {
                "platform": "facebook",
                "selector": "div[role='button']",
                "textContains": "next",
                "timeoutMs": 120000,
                "waitForUpload": True,
                "singleClick": True,
            },
        )
        return
    except Exception:
        pass

    # Fallback: wait for explicit 100% badge in dialog if present.
    _wait_for_query_match(
        executor,
        "facebook",
        "div[role='dialog'] span:has-text('100%'), div[role='dialog'] div:has-text('100%')",
        timeout_s=120.0,
        poll_s=0.5,
    )

def _attempt_fill_description(executor: BaseExecutor, description: str) -> bool:
    selectors = [
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='What\'s on your mind' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='What\'s on your mind' i]",
        "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='What\'s on your mind' i]",
        "div[contenteditable='true'][role='textbox'][aria-label*='What\'s on your mind' i]",
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='Describe your reel' i]",
        "[contenteditable='true'][aria-placeholder*='What\'s on your mind' i]",
        "[contenteditable='true'][aria-placeholder*='Describe your reel' i]",
        "div[contenteditable='true'][role='textbox']",
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
            _best_effort_click(executor, "facebook", selector, timeout_ms=5000)

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


def run(executor: BaseExecutor, video_path: str, caption: str, title: str, platform_url: str = "") -> dict[str, Any]:
    executor.run(
        "platform.open",
        {
            "platform": "facebook",
            "url": _safe_facebook_url(platform_url),
            "reuseTab": True,
        },
    )
    executor.run("platform.ensure_logged_in", {"platform": "facebook"})

    # User-homepage post flow: open composer from profile/home surface.
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='Create post' i]", timeout_ms=10000)
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='What\'s on your mind' i]", timeout_ms=10000)
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="create post")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="what's on your mind")

    # Open media picker in create-post dialog/page.
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='Photo/video' i]", timeout_ms=10000)
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="photo/video")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="add photos/videos")

    executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})

    _sleep_between_actions(executor, "facebook upload progress")
    _wait_for_facebook_upload_ready(executor)

    # Ensure the post composer textbox is visible before trying to fill caption + hashtags.
    composer_ready = _wait_for_query_match(
        executor,
        "facebook",
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='What\'s on your mind' i], div[contenteditable='true'][role='textbox'][aria-placeholder*='What\'s on your mind' i]",
        timeout_s=45.0,
        poll_s=0.4,
    )
    if not composer_ready:
        raise RuntimeError("Facebook post composer did not appear after upload")

    _sleep_between_actions(executor, "facebook composer description fill")
    description_ok = _attempt_fill_description(executor, caption)
    if not description_ok:
        raise RuntimeError(
            "Facebook description fill failed; aborting before submit to avoid posting without caption"
        )

    _sleep_between_actions(executor, "facebook next + post")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="next")
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']", timeout_ms=10000)
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=12000, text_contains="post")
    _best_effort_click(executor, "facebook", "div[aria-label='Post'][role='button']", timeout_ms=12000)

    executor.run("post.submit", {"platform": "facebook"})
    return executor.run("post.status", {"platform": "facebook"})
