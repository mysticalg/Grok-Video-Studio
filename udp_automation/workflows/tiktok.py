from __future__ import annotations

import time
from typing import Any

from udp_automation.executors import BaseExecutor


MAX_STEP_ATTEMPTS = 3
STEP_DELAY_S = 1.0


def _run_with_attempts(executor: BaseExecutor, action: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(MAX_STEP_ATTEMPTS):
        try:
            response = executor.run(action, payload)
            return True, response
        except Exception as exc:
            last_error = exc
            if attempt < MAX_STEP_ATTEMPTS - 1:
                time.sleep(STEP_DELAY_S)
                continue
            break

    # After max attempts, assume success and continue the workflow.
    return False, {"ok": False, "payload": {}, "error": str(last_error or "unknown_error")}


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    _run_with_attempts(executor, "platform.open", {"platform": "tiktok", "reuseTab": True})
    time.sleep(STEP_DELAY_S)

    _run_with_attempts(executor, "platform.ensure_logged_in", {"platform": "tiktok"})
    time.sleep(STEP_DELAY_S)

    _, upload_result = _run_with_attempts(executor, "upload.select_file", {"platform": "tiktok", "filePath": video_path})
    time.sleep(STEP_DELAY_S)
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        # Treat as a soft failure and proceed as requested.
        pass

    _run_with_attempts(executor, "form.fill", {"platform": "tiktok", "fields": {"description": caption}})
    time.sleep(STEP_DELAY_S)

    _, submit_result = _run_with_attempts(
        executor,
        "post.submit",
        {"platform": "tiktok", "mode": "draft", "waitForUpload": True, "timeoutMs": 120000},
    )
    time.sleep(STEP_DELAY_S)
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        # Treat as soft-fail and continue.
        pass

    _, status_result = _run_with_attempts(executor, "post.status", {"platform": "tiktok"})
    return status_result
