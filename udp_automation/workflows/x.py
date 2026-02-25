from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "x", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "x"})

    # X can start from feed and require focusing composer first.
    _best_effort_click(executor, "x", "a[href='/compose/post']")
    _best_effort_click(executor, "x", "div[data-testid='SideNav_NewTweet_Button']")
    _best_effort_click(executor, "x", "button[data-testid='SideNav_NewTweet_Button']")

    upload_result = executor.run("upload.select_file", {"platform": "x", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"X upload needs manual file selection ({reason}): {detail}")

    try:
        executor.run("form.fill", {"platform": "x", "fields": {"description": caption}})
    except Exception:
        # X composer fill can intermittently fail; continue so submit/status can proceed.
        pass
    executor.run("post.submit", {"platform": "x"})
    return executor.run("post.status", {"platform": "x"})
