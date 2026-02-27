from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def _must_click(executor: BaseExecutor, selector: str, timeout_ms: int = 30000) -> None:
    result = executor.run(
        "dom.click",
        {
            "platform": "tiktok",
            "selector": selector,
            "timeoutMs": timeout_ms,
        },
    )
    payload = result.get("payload") or {}
    if payload.get("clicked") is False:
        raise RuntimeError(f"TikTok element was not found/clicked for selector: {selector}")


def _must_type(executor: BaseExecutor, selector: str, value: str) -> None:
    result = executor.run(
        "dom.type",
        {
            "platform": "tiktok",
            "selector": selector,
            "value": value,
        },
    )
    payload = result.get("payload") or {}
    if payload.get("typed") is False:
        raise RuntimeError(f"TikTok element was not found/typed for selector: {selector}")


def run(executor: BaseExecutor, video_path: str, caption: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    publish_mode = str(opts.get("publish_mode") or "draft").strip().lower()
    if publish_mode not in {"draft", "post"}:
        publish_mode = "draft"
    add_text = bool(opts.get("add_text_overlay"))
    add_music = bool(opts.get("add_music"))
    music_query = str(opts.get("music_query") or "").strip()

    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})

    upload_result = executor.run("upload.select_file", {"platform": "tiktok", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"TikTok upload needs manual file selection ({reason}): {detail}")

    executor.run("form.fill", {"platform": "tiktok", "fields": {"description": caption}})

    if add_text or add_music:
        _must_click(executor, "button[data-button-name='edit'], button.editor-entrance", timeout_ms=120000)

    if add_text:
        _must_click(executor, "div[data-name='AddTextPresetPanel']", timeout_ms=60000)
        _must_click(executor, "button.AddTextPanel__addTextBasicButton", timeout_ms=60000)
        _must_type(executor, "textarea[name='content']", caption)

    if add_music and music_query:
        _must_click(executor, "div[data-name='MusicPanel']", timeout_ms=60000)
        _must_type(executor, "input[placeholder='Search sounds']", music_query)
        _must_click(executor, ".search-bar-container .search-icon, button[aria-label*='search' i]", timeout_ms=30000)
        random_track_result = executor.run(
            "dom.click_random",
            {
                "platform": "tiktok",
                "selector": "div.MusicPanelMusicItem__operation button[role='button']",
                "timeoutMs": 60000,
            },
        )
        random_track_payload = random_track_result.get("payload") or {}
        if random_track_payload.get("clicked") is False:
            raise RuntimeError("TikTok random music track button was not found/clicked")

    if add_text or (add_music and music_query):
        _must_click(executor, "button.Button__root--type-primary", timeout_ms=60000)

    submit_result = executor.run(
        "post.submit",
        {
            "platform": "tiktok",
            "mode": publish_mode,
            "waitForUpload": True,
            "timeoutMs": 120000,
        },
    )
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        raise RuntimeError(f"TikTok {publish_mode} button was not found/clicked")

    return executor.run("post.status", {"platform": "tiktok"})
