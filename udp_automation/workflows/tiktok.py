from __future__ import annotations

import time
from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    step_delay_s = 1.0

    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    time.sleep(step_delay_s)

    # Give the page a moment to stabilize before auth-state checks.
    time.sleep(step_delay_s)
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})
    time.sleep(step_delay_s)

    upload_result = executor.run("upload.select_file", {"platform": "tiktok", "filePath": video_path})
    time.sleep(step_delay_s)
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"TikTok upload needs manual file selection ({reason}): {detail}")

    # Best-effort only: if caption fill fails, continue to submission step.
    try:
        executor.run("form.fill", {"platform": "tiktok", "fields": {"description": caption}})
    except Exception:
        pass
    time.sleep(step_delay_s)

    submit_result = executor.run("post.submit", {"platform": "tiktok", "mode": "draft", "waitForUpload": True, "timeoutMs": 120000})
    time.sleep(step_delay_s)
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        raise RuntimeError("TikTok post button was not found/clicked")

    return executor.run("post.status", {"platform": "tiktok"})
