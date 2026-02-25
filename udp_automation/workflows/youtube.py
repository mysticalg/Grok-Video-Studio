from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def _best_effort_type(executor: BaseExecutor, platform: str, selector: str, value: str) -> bool:
    try:
        resp = executor.run("dom.type", {"platform": platform, "selector": selector, "value": value})
        payload = (resp or {}).get("payload") or {}
        return bool(payload.get("typed"))
    except Exception:
        return False


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

    # Prefer explicit textbox typing over generic form.fill to avoid long UDP waits.
    title_ok = True
    desc_ok = True
    if title:
        title_ok = _best_effort_type(
            executor,
            "youtube",
            "div#textbox[contenteditable='true'][role='textbox'][aria-label*='Add a title' i]",
            title,
        )
    if description:
        desc_ok = _best_effort_type(
            executor,
            "youtube",
            "div#textbox[contenteditable='true'][role='textbox'][aria-label*='Tell viewers about your video' i]",
            description,
        )

    if not (title_ok and desc_ok):
        _best_effort_log_note(executor, f"youtube dom.type fill partial title_ok={title_ok} description_ok={desc_ok}")

    # Prefer service-side CDP sequence to avoid extension dom.click hangs.
    try:
        executor.run("youtube.publish_steps", {"platform": "youtube"})
    except Exception as exc:
        _best_effort_log_note(executor, f"youtube.publish_steps failed: {exc}")

    return executor.run("post.status", {"platform": "youtube"})
