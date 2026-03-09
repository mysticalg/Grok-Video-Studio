#     *    
#    * *   
#   * * *  
#  *  *  * 
# *********
#  *  *  * 
#   * * *  
#    * *   
#     *    

"""Youtube module for Grok Video Studio automation flows."""

from __future__ import annotations

import os
import time
from typing import Any

from udp_automation.executors import BaseExecutor


YOUTUBE_STEP_DELAY_SECONDS = max(0.0, float(os.getenv("GROK_YOUTUBE_STEP_DELAY_SECONDS", "1.2")))
YOUTUBE_FORM_FILL_ATTEMPTS = max(1, int(os.getenv("GROK_YOUTUBE_FORM_FILL_ATTEMPTS", "3")))
YOUTUBE_PUBLISH_ATTEMPTS = max(1, int(os.getenv("GROK_YOUTUBE_PUBLISH_ATTEMPTS", "2")))


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str, *, timeout_ms: int) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": timeout_ms})
    except Exception:
        return


def _best_effort_log(executor: BaseExecutor, action: str, status: str, note: str) -> None:
    try:
        logger = getattr(executor, "_log", None)
        if callable(logger):
            logger(action, status, note)
    except Exception:
        return


def _best_effort_log_note(executor: BaseExecutor, note: str) -> None:
    _best_effort_log(executor, "youtube.workflow", "info", note)


def _sleep_between_actions(executor: BaseExecutor, reason: str, *, step_delay_seconds: float) -> None:
    if step_delay_seconds <= 0:
        return
    _best_effort_log_note(
        executor,
        f"youtube workflow delay {step_delay_seconds:.1f}s before {reason}",
    )
    time.sleep(step_delay_seconds)


def _attempt_form_fill(executor: BaseExecutor, title: str, description: str, *, form_fill_attempts: int, step_delay_seconds: float) -> bool:
    for attempt in range(1, form_fill_attempts + 1):
        _best_effort_log_note(
            executor,
            (
                f"youtube form.fill attempt {attempt}/{form_fill_attempts} "
                f"title_len={len((title or '').strip())} description_len={len((description or '').strip())}"
            ),
        )
        try:
            executor.run("form.fill", {"platform": "youtube", "fields": {"title": title, "description": description}})
            _best_effort_log_note(executor, f"youtube form.fill attempt {attempt}/{form_fill_attempts} succeeded")
            return True
        except Exception as exc:
            _best_effort_log(executor, "form.fill", "warning", f"youtube form.fill attempt {attempt} failed: {exc}")
            if attempt < form_fill_attempts:
                _sleep_between_actions(executor, f"youtube form.fill retry {attempt + 1}", step_delay_seconds=step_delay_seconds)
    return False


def _attempt_publish_steps(executor: BaseExecutor, *, publish_attempts: int, step_delay_seconds: float) -> bool:
    for attempt in range(1, publish_attempts + 1):
        _best_effort_log_note(executor, f"youtube.publish_steps attempt {attempt}/{publish_attempts}")
        try:
            executor.run("youtube.publish_steps", {"platform": "youtube"})
            _best_effort_log_note(executor, f"youtube.publish_steps attempt {attempt}/{publish_attempts} succeeded")
            return True
        except Exception as exc:
            _best_effort_log(executor, "youtube.publish_steps", "warning", f"youtube.publish_steps attempt {attempt} failed: {exc}")
            if attempt < publish_attempts:
                _sleep_between_actions(executor, f"youtube.publish_steps retry {attempt + 1}", step_delay_seconds=step_delay_seconds)
    return False


def run(executor: BaseExecutor, video_path: str, title: str, description: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    step_delay_seconds = max(0.0, float((opts.get("youtube_step_delay_ms") if opts.get("youtube_step_delay_ms") is not None else int(YOUTUBE_STEP_DELAY_SECONDS * 1000)) ) / 1000.0)
    form_fill_attempts = max(1, int(opts.get("youtube_form_fill_attempts") or YOUTUBE_FORM_FILL_ATTEMPTS))
    publish_attempts = max(1, int(opts.get("youtube_publish_attempts") or YOUTUBE_PUBLISH_ATTEMPTS))
    click_timeout_ms = max(1000, int(opts.get("youtube_click_timeout_ms") or 8000))

    executor.run("platform.open", {"platform": "youtube", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "youtube"})

    # YouTube Studio often requires opening the Create menu before the upload file input exists.
    _best_effort_click(executor, "youtube", "button[aria-label='Create']", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "youtube", "button[aria-label*='Create' i]", timeout_ms=click_timeout_ms)
    _sleep_between_actions(executor, "youtube upload-menu selection", step_delay_seconds=step_delay_seconds)
    _best_effort_click(executor, "youtube", "tp-yt-paper-item[test-id='upload']", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "youtube", "tp-yt-paper-item#text-item-0[test-id='upload']", timeout_ms=click_timeout_ms)

    executor.run("upload.select_file", {"platform": "youtube", "filePath": video_path})
    _sleep_between_actions(executor, "youtube metadata form.fill", step_delay_seconds=step_delay_seconds)

    form_fill_ok = _attempt_form_fill(executor, title, description, form_fill_attempts=form_fill_attempts, step_delay_seconds=step_delay_seconds)
    if not form_fill_ok:
        _best_effort_log(executor, "form.fill", "warning", "youtube form.fill exhausted all attempts")

    _sleep_between_actions(executor, "youtube publish steps", step_delay_seconds=step_delay_seconds)

    # Prefer service-side CDP sequence to avoid extension dom.click hangs.
    publish_ok = _attempt_publish_steps(executor, publish_attempts=publish_attempts, step_delay_seconds=step_delay_seconds)
    if not publish_ok:
        _best_effort_log(executor, "youtube.publish_steps", "warning", "youtube.publish_steps exhausted all attempts")

    return executor.run("post.status", {"platform": "youtube"})
