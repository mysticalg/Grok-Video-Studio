from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": 8000})
    except Exception:
        return


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "instagram", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "instagram"})

    # Instagram may require entering the reel composer before file input appears.
    _best_effort_click(executor, "instagram", "a[href*='create/reel']")
    _best_effort_click(executor, "instagram", "button[aria-label*='Select from computer' i]")
    _best_effort_click(executor, "instagram", "div[role='button'][aria-label*='Select from computer' i]")

    upload_result = executor.run("upload.select_file", {"platform": "instagram", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"Instagram upload needs manual file selection ({reason}): {detail}")

    executor.run("form.fill", {"platform": "instagram", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "instagram"})
    return executor.run("post.status", {"platform": "instagram"})
