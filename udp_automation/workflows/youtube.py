from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return




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

    try:
        executor.run("form.fill", {"platform": "youtube", "fields": {"title": title, "description": description}})
    except Exception as exc:
        _best_effort_log_note(executor, f"youtube form.fill skipped due to error: {exc}")

    # Prefer service-side CDP sequence to avoid extension dom.click hangs.
    try:
        executor.run("youtube.publish_steps", {"platform": "youtube"})
    except Exception:
        # Fallback to extension clicks if CDP publish sequence is unavailable.
        for _ in range(3):
            _best_effort_click(executor, "youtube", "button[aria-label='Next']")
            _best_effort_click(executor, "youtube", "button[aria-label*='Next' i]")

        _best_effort_click(executor, "youtube", "button[aria-label='Save']")
        _best_effort_click(executor, "youtube", "button[aria-label*='Save' i]")

    return executor.run("post.status", {"platform": "youtube"})
