from __future__ import annotations

import os
import time
from typing import Any, Callable

from udp_automation.executors import BaseExecutor


FACEBOOK_STEP_DELAY_SECONDS = max(0.0, float(os.getenv("GROK_FACEBOOK_STEP_DELAY_SECONDS", "1.0")))
FACEBOOK_FORM_FILL_ATTEMPTS = max(1, int(os.getenv("GROK_FACEBOOK_FORM_FILL_ATTEMPTS", "3")))


LogFn = Callable[[str], None]


def _emit_progress(log_fn: LogFn | None, message: str) -> None:
    if callable(log_fn):
        try:
            log_fn(str(message))
        except Exception:
            return


def _best_effort_log(executor: BaseExecutor, action: str, status: str, note: str) -> None:
    try:
        logger = getattr(executor, "_log", None)
        if callable(logger):
            logger(action, status, note)
    except Exception:
        return


def _best_effort_log_note(executor: BaseExecutor, note: str, *, log_fn: LogFn | None = None) -> None:
    _best_effort_log(executor, "facebook.workflow", "info", note)
    _emit_progress(log_fn, note)


def _best_effort_click(
    executor: BaseExecutor,
    platform: str,
    selector: str,
    *,
    timeout_ms: int = 8000,
    text_contains: str = "",
    log_fn: LogFn | None = None,
    step_name: str = "dom.click",
) -> None:
    payload: dict[str, Any] = {"platform": platform, "selector": selector, "timeoutMs": timeout_ms}
    if text_contains:
        payload["textContains"] = text_contains

    _emit_progress(log_fn, f"{step_name}: selector={selector} text_contains={text_contains or '-'} timeoutMs={timeout_ms}")
    try:
        result = executor.run("dom.click", payload)
        _emit_progress(log_fn, f"{step_name}: ok payload={result.get('payload') or {}}")
    except Exception as exc:
        _emit_progress(log_fn, f"{step_name}: miss error={exc}")


def _sleep_between_actions(executor: BaseExecutor, reason: str, *, log_fn: LogFn | None = None) -> None:
    if FACEBOOK_STEP_DELAY_SECONDS <= 0:
        return
    note = f"facebook workflow delay {FACEBOOK_STEP_DELAY_SECONDS:.1f}s before {reason}"
    _best_effort_log_note(executor, note, log_fn=log_fn)
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
    log_fn: LogFn | None = None,
    label: str = "dom.query",
) -> bool:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    poll_s = max(0.1, float(poll_s))
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        try:
            result = executor.run("dom.query", {"platform": platform, "selector": selector})
            payload = result.get("payload") or {}
            count = int(payload.get("count") or 0)
            _emit_progress(log_fn, f"{label}: attempt={attempts} selector={selector} count={count}")
            if count > 0:
                return True
        except Exception as exc:
            _emit_progress(log_fn, f"{label}: attempt={attempts} selector={selector} error={exc}")
        time.sleep(poll_s)
    _emit_progress(log_fn, f"{label}: timeout selector={selector} after_attempts={attempts}")
    return False


def _wait_for_facebook_upload_ready(executor: BaseExecutor, *, log_fn: LogFn | None = None) -> None:
    _emit_progress(log_fn, "facebook upload readiness: waiting for Next click with waitForUpload=true")
    try:
        result = executor.run(
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
        _emit_progress(log_fn, f"facebook upload readiness: waitForUpload gate passed payload={result.get('payload') or {}}")
        return
    except Exception as exc:
        _emit_progress(log_fn, f"facebook upload readiness: waitForUpload gate failed error={exc}")

    _wait_for_query_match(
        executor,
        "facebook",
        "div[role='dialog'] span:has-text('100%'), div[role='dialog'] div:has-text('100%')",
        timeout_s=120.0,
        poll_s=0.5,
        log_fn=log_fn,
        label="facebook upload readiness 100%",
    )


def _attempt_fill_description(executor: BaseExecutor, description: str, *, log_fn: LogFn | None = None) -> bool:
    selectors = [
        "div.xzsf02u[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='mind' i]",
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='mind' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='mind' i]",
        "div[contenteditable='true'][data-lexical-editor='true'][aria-placeholder*='mind' i]",
        "div[contenteditable='true'][role='textbox'][aria-label*='mind' i]",
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='What\'s on your mind' i]",
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='What\'s on your mind' i]",
        "[contenteditable='true'][aria-placeholder*='mind' i]",
        "div[contenteditable='true'][role='textbox']",
    ]

    value = str(description or "").strip()
    if not value:
        _best_effort_log_note(executor, "facebook description is empty; skipping description fill", log_fn=log_fn)
        return True

    for attempt in range(1, FACEBOOK_FORM_FILL_ATTEMPTS + 1):
        _best_effort_log_note(
            executor,
            f"facebook description attempt {attempt}/{FACEBOOK_FORM_FILL_ATTEMPTS}: begin len={len(value)}",
            log_fn=log_fn,
        )

        # Focus attempts for each selector.
        for selector in selectors:
            _best_effort_click(
                executor,
                "facebook",
                selector,
                timeout_ms=5000,
                log_fn=log_fn,
                step_name=f"facebook description attempt {attempt} focus",
            )

        # Primary path: platform-aware form.fill.
        try:
            _emit_progress(log_fn, f"facebook description attempt {attempt}: form.fill start")
            response = executor.run("form.fill", {"platform": "facebook", "fields": {"description": value}})
            payload = response.get("payload") or {}
            fill_ok = bool(payload.get("description") is True or payload.get("description") == 1)
            _emit_progress(log_fn, f"facebook description attempt {attempt}: form.fill payload={payload}")
            if fill_ok:
                _best_effort_log_note(
                    executor,
                    f"facebook description attempt {attempt}: form.fill success",
                    log_fn=log_fn,
                )
                return True
        except Exception as exc:
            _emit_progress(log_fn, f"facebook description attempt {attempt}: form.fill error={exc}")

        # Explicit type attempts into the exact textbox candidates.
        for selector in selectors:
            try:
                _emit_progress(log_fn, f"facebook description attempt {attempt}: dom.type start selector={selector}")
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
                _emit_progress(log_fn, f"facebook description attempt {attempt}: dom.type payload={type_payload} selector={selector}")
                if typed_ok:
                    _best_effort_log_note(
                        executor,
                        f"facebook description attempt {attempt}: dom.type success selector={selector}",
                        log_fn=log_fn,
                    )
                    return True
            except Exception as exc:
                _emit_progress(log_fn, f"facebook description attempt {attempt}: dom.type error={exc} selector={selector}")

        if attempt < FACEBOOK_FORM_FILL_ATTEMPTS:
            _sleep_between_actions(executor, f"facebook description fill retry {attempt + 1}", log_fn=log_fn)

    _best_effort_log_note(executor, "facebook description attempts exhausted", log_fn=log_fn)
    return False


def run(
    executor: BaseExecutor,
    video_path: str,
    caption: str,
    title: str,
    platform_url: str = "",
    log_fn: LogFn | None = None,
) -> dict[str, Any]:
    _emit_progress(log_fn, f"facebook workflow start: platform_url={platform_url or '-'}")
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
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='Create post' i]", timeout_ms=10000, log_fn=log_fn, step_name="facebook composer open")
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='What\'s on your mind' i]", timeout_ms=10000, log_fn=log_fn, step_name="facebook composer open")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="create post", log_fn=log_fn, step_name="facebook composer open")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="what's on your mind", log_fn=log_fn, step_name="facebook composer open")

    # Open media picker in create-post dialog/page.
    _best_effort_click(executor, "facebook", "div[role='button'][aria-label*='Photo/video' i]", timeout_ms=10000, log_fn=log_fn, step_name="facebook picker")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="photo/video", log_fn=log_fn, step_name="facebook picker")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="add photos/videos", log_fn=log_fn, step_name="facebook picker")

    _emit_progress(log_fn, f"facebook upload.select_file start: {video_path}")
    upload_result = executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})
    _emit_progress(log_fn, f"facebook upload.select_file done payload={upload_result.get('payload') or {}}")

    _sleep_between_actions(executor, "facebook upload progress", log_fn=log_fn)
    _wait_for_facebook_upload_ready(executor, log_fn=log_fn)

    # Ensure the post composer textbox is visible before trying to fill caption + hashtags.
    composer_ready = _wait_for_query_match(
        executor,
        "facebook",
        "div.xzsf02u[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='mind' i], "
        "div[contenteditable='true'][role='textbox'][data-lexical-editor='true'][aria-placeholder*='mind' i], "
        "div[contenteditable='true'][role='textbox'][aria-placeholder*='mind' i]",
        timeout_s=45.0,
        poll_s=0.4,
        log_fn=log_fn,
        label="facebook composer wait",
    )
    if not composer_ready:
        raise RuntimeError("Facebook post composer did not appear after upload")

    _sleep_between_actions(executor, "facebook composer description fill", log_fn=log_fn)
    description_ok = _attempt_fill_description(executor, caption, log_fn=log_fn)
    if not description_ok:
        raise RuntimeError(
            "Facebook description fill failed; aborting before submit to avoid posting without caption"
        )

    _sleep_between_actions(executor, "facebook next + post", log_fn=log_fn)
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=10000, text_contains="next", log_fn=log_fn, step_name="facebook publish")
    _best_effort_click(executor, "facebook", "div[aria-label='Next'][role='button']", timeout_ms=10000, log_fn=log_fn, step_name="facebook publish")
    _best_effort_click(executor, "facebook", "div[role='button']", timeout_ms=12000, text_contains="post", log_fn=log_fn, step_name="facebook publish")
    _best_effort_click(executor, "facebook", "div[aria-label='Post'][role='button']", timeout_ms=12000, log_fn=log_fn, step_name="facebook publish")

    _emit_progress(log_fn, "facebook post.submit start")
    executor.run("post.submit", {"platform": "facebook"})
    status = executor.run("post.status", {"platform": "facebook"})
    _emit_progress(log_fn, f"facebook post.status payload={status.get('payload') or {}}")
    return status
