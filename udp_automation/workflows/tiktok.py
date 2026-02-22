from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})

    upload_result = executor.run("upload.select_file", {"platform": "tiktok", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"TikTok upload needs manual file selection ({reason}): {detail}")

    executor.run("form.fill", {"platform": "tiktok", "fields": {"description": caption}})

    submit_result = executor.run("post.submit", {"platform": "tiktok"})
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        raise RuntimeError("TikTok post button was not found/clicked")

    return executor.run("post.status", {"platform": "tiktok"})
