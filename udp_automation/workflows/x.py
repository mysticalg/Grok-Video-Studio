from __future__ import annotations

import time
from typing import Any

from udp_automation.executors import BaseExecutor


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str, *, timeout_ms: int) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": timeout_ms})
    except Exception:
        return


def run(executor: BaseExecutor, video_path: str, caption: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    compose_click_timeout_ms = max(1000, int(opts.get("x_compose_click_timeout_ms") or 8000))
    description_fill_attempts = max(1, int(opts.get("x_description_fill_attempts") or 3))
    description_fill_retry_delay_ms = max(0, int(opts.get("x_description_fill_retry_delay_ms") or 500))

    executor.run("platform.open", {"platform": "x", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "x"})

    # Keep the workflow in a single browser tab.
    #
    # Clicking the compose nav link can spawn a second tab/popout in some X sessions,
    # which breaks relay-backed workflows that are bound to the original tab.
    # `platform.open` already targets /compose/post, so only try in-page compose
    # buttons as fallback to focus the existing composer surface.
    _best_effort_click(executor, "x", "div[data-testid='SideNav_NewTweet_Button']", timeout_ms=compose_click_timeout_ms)
    _best_effort_click(executor, "x", "button[data-testid='SideNav_NewTweet_Button']", timeout_ms=compose_click_timeout_ms)

    upload_result = executor.run("upload.select_file", {"platform": "x", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"X upload needs manual file selection ({reason}): {detail}")

    if caption.strip():
        fill_ok = False
        last_fill_error = ""
        for attempt in range(1, description_fill_attempts + 1):
            try:
                fill_result = executor.run("form.fill", {"platform": "x", "fields": {"description": caption}})
                fill_payload = fill_result.get("payload") or {}
                fill_ok = bool(fill_payload.get("description") is True or fill_payload.get("description") == 1)
                if fill_ok:
                    break
            except Exception as exc:
                last_fill_error = str(exc)
            if not fill_ok and attempt < description_fill_attempts and description_fill_retry_delay_ms > 0:
                time.sleep(description_fill_retry_delay_ms / 1000.0)

        if not fill_ok:
            raise RuntimeError(
                "X composer description fill failed"
                + (f": {last_fill_error}" if last_fill_error else "")
            )
    executor.run("post.submit", {"platform": "x"})
    return executor.run("post.status", {"platform": "x"})
