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

    submit_result = executor.run("post.submit", {"platform": "tiktok", "mode": "draft", "waitForUpload": True, "timeoutMs": 120000})
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        raise RuntimeError("TikTok post button was not found/clicked")

    draft_open_result = executor.run(
        "dom.click",
        {
            "platform": "tiktok",
            "selector": "button[data-tt='components_DraftCells_Clickable']",
            "timeoutMs": 120000,
        },
    )
    draft_open_payload = draft_open_result.get("payload") or {}
    if draft_open_payload.get("clicked") is False:
        raise RuntimeError("TikTok draft entry button was not found/clicked")

    edit_video_result = executor.run(
        "dom.click",
        {
            "platform": "tiktok",
            "selector": "button.TUXButton.TUXButton--default.TUXButton--medium.TUXButton--secondary",
            "timeoutMs": 120000,
        },
    )
    edit_video_payload = edit_video_result.get("payload") or {}
    if edit_video_payload.get("clicked") is False:
        raise RuntimeError("TikTok 'Edit video' button was not found/clicked")

    search_fill_result = executor.run(
        "dom.type",
        {
            "platform": "tiktok",
            "selector": "input.search-bar-input",
            "value": "Infinite Dimensions",
        },
    )
    search_fill_payload = search_fill_result.get("payload") or {}
    if search_fill_payload.get("typed") is False:
        raise RuntimeError("TikTok search input was not found/typed")

    search_submit_result = executor.run(
        "dom.click",
        {
            "platform": "tiktok",
            "selector": ".search-bar-container .search-icon",
            "timeoutMs": 30000,
        },
    )
    search_submit_payload = search_submit_result.get("payload") or {}
    if search_submit_payload.get("clicked") is False:
        raise RuntimeError("TikTok search submit button was not found/clicked")

    return executor.run("post.status", {"platform": "tiktok"})
