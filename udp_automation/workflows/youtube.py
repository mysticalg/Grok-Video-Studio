from __future__ import annotations

import os
import time
from typing import Any

from udp_automation.executors import BaseExecutor


YOUTUBE_STEP_DELAY_SECONDS = max(0.0, float(os.getenv("GROK_YOUTUBE_STEP_DELAY_SECONDS", "1.2")))
YOUTUBE_FORM_FILL_ATTEMPTS = max(1, int(os.getenv("GROK_YOUTUBE_FORM_FILL_ATTEMPTS", "3")))
YOUTUBE_PUBLISH_ATTEMPTS = max(1, int(os.getenv("GROK_YOUTUBE_PUBLISH_ATTEMPTS", "2")))


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
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


def _sleep_between_actions(executor: BaseExecutor, reason: str) -> None:
    if YOUTUBE_STEP_DELAY_SECONDS <= 0:
        return
    _best_effort_log_note(
        executor,
        f"youtube workflow delay {YOUTUBE_STEP_DELAY_SECONDS:.1f}s before {reason}",
    )
    time.sleep(YOUTUBE_STEP_DELAY_SECONDS)


def _attempt_form_fill(executor: BaseExecutor, title: str, description: str) -> bool:
    for attempt in range(1, YOUTUBE_FORM_FILL_ATTEMPTS + 1):
        _best_effort_log_note(
            executor,
            (
                f"youtube form.fill attempt {attempt}/{YOUTUBE_FORM_FILL_ATTEMPTS} "
                f"title_len={len((title or '').strip())} description_len={len((description or '').strip())}"
            ),
        )
        try:
            executor.run("form.fill", {"platform": "youtube", "fields": {"title": title, "description": description}})
            _best_effort_log_note(executor, f"youtube form.fill attempt {attempt}/{YOUTUBE_FORM_FILL_ATTEMPTS} succeeded")
            return True
        except Exception as exc:
            _best_effort_log(executor, "form.fill", "warning", f"youtube form.fill attempt {attempt} failed: {exc}")
            if attempt < YOUTUBE_FORM_FILL_ATTEMPTS:
                _sleep_between_actions(executor, f"youtube form.fill retry {attempt + 1}")
    return False


def _attempt_publish_steps(executor: BaseExecutor) -> bool:
    for attempt in range(1, YOUTUBE_PUBLISH_ATTEMPTS + 1):
        _best_effort_log_note(executor, f"youtube.publish_steps attempt {attempt}/{YOUTUBE_PUBLISH_ATTEMPTS}")
        try:
            executor.run("youtube.publish_steps", {"platform": "youtube"})
            _best_effort_log_note(executor, f"youtube.publish_steps attempt {attempt}/{YOUTUBE_PUBLISH_ATTEMPTS} succeeded")
            return True
        except Exception as exc:
            _best_effort_log(executor, "youtube.publish_steps", "warning", f"youtube.publish_steps attempt {attempt} failed: {exc}")
            if attempt < YOUTUBE_PUBLISH_ATTEMPTS:
                _sleep_between_actions(executor, f"youtube.publish_steps retry {attempt + 1}")
    return False


def run(executor: BaseExecutor, video_path: str, title: str, description: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "youtube", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "youtube"})

    # YouTube Studio often requires opening the Create menu before the upload file input exists.
    _best_effort_click(executor, "youtube", "button[aria-label='Create']")
    _best_effort_click(executor, "youtube", "button[aria-label*='Create' i]")
    _sleep_between_actions(executor, "youtube upload-menu selection")
    _best_effort_click(executor, "youtube", "tp-yt-paper-item[test-id='upload']")
    _best_effort_click(executor, "youtube", "tp-yt-paper-item#text-item-0[test-id='upload']")

    executor.run("upload.select_file", {"platform": "youtube", "filePath": video_path})
    _sleep_between_actions(executor, "youtube metadata form.fill")

    form_fill_ok = _attempt_form_fill(executor, title, description)
    if not form_fill_ok:
        _best_effort_log(executor, "form.fill", "warning", "youtube form.fill exhausted all attempts")

    _sleep_between_actions(executor, "youtube publish steps")

    # Prefer service-side CDP sequence to avoid extension dom.click hangs.
    publish_ok = _attempt_publish_steps(executor)
    if not publish_ok:
        _best_effort_log(executor, "youtube.publish_steps", "warning", "youtube.publish_steps exhausted all attempts")

    return executor.run("post.status", {"platform": "youtube"})
