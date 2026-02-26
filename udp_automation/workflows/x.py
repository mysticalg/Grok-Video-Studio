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

    # Keep the workflow in a single browser tab.
    #
    # Clicking the compose nav link can spawn a second tab/popout in some X sessions,
    # which breaks relay-backed workflows that are bound to the original tab.
    # `platform.open` already targets /compose/post, so only try in-page compose
    # buttons as fallback to focus the existing composer surface.
    _best_effort_click(executor, "x", "div[data-testid='SideNav_NewTweet_Button']")
    _best_effort_click(executor, "x", "button[data-testid='SideNav_NewTweet_Button']")

    upload_result = executor.run("upload.select_file", {"platform": "x", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"X upload needs manual file selection ({reason}): {detail}")

    if caption.strip():
        fill_ok = False
        last_fill_error = ""
        for _ in range(3):
            try:
                fill_result = executor.run("form.fill", {"platform": "x", "fields": {"description": caption}})
                fill_payload = fill_result.get("payload") or {}
                fill_ok = bool(fill_payload.get("description") is True or fill_payload.get("description") == 1)
                if fill_ok:
                    break
            except Exception as exc:
                last_fill_error = str(exc)

        if not fill_ok:
            raise RuntimeError(
                "X composer description fill failed"
                + (f": {last_fill_error}" if last_fill_error else "")
            )
    executor.run("post.submit", {"platform": "x"})
    return executor.run("post.status", {"platform": "x"})
