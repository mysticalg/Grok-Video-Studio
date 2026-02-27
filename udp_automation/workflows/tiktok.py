from __future__ import annotations

import re
from typing import Any, Callable

from udp_automation.executors import BaseExecutor


LogFn = Callable[[str], None]


def _log(log_fn: LogFn | None, message: str) -> None:
    if callable(log_fn):
        try:
            log_fn(message)
        except Exception:
            pass


def _must_click(executor: BaseExecutor, selector: str, timeout_ms: int = 30000, *, step: str = "", log_fn: LogFn | None = None) -> dict[str, Any]:
    _log(log_fn, f"{step or 'click'}: selector={selector} timeout_ms={timeout_ms}")
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
        reason = payload.get("reason") or "unknown"
        raise RuntimeError(f"TikTok {step or 'click'} failed (selector={selector}, reason={reason})")
    _log(log_fn, f"{step or 'click'}: ok payload={payload}")
    return payload


def _must_type(executor: BaseExecutor, selector: str, value: str, *, step: str = "", log_fn: LogFn | None = None) -> None:
    _log(log_fn, f"{step or 'type'}: selector={selector} chars={len(value)}")
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
        raise RuntimeError(f"TikTok {step or 'type'} failed (selector={selector})")
    _log(log_fn, f"{step or 'type'}: ok payload={payload}")


def _overlay_text(opts: dict[str, Any], caption: str) -> str:
    configured = str(opts.get("text_overlay") or "").strip()
    raw = configured or str(caption or "").strip()
    without_tags = re.sub(r"(^|\s)#\w+", " ", raw)
    return " ".join(without_tags.split()).strip()


def run(executor: BaseExecutor, video_path: str, caption: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    log_fn = opts.get("_log_callback") if callable(opts.get("_log_callback")) else None
    publish_mode = str(opts.get("publish_mode") or "draft").strip().lower()
    if publish_mode not in {"draft", "post"}:
        publish_mode = "draft"
    add_text = bool(opts.get("add_text_overlay"))
    add_music = bool(opts.get("add_music"))
    music_query = str(opts.get("music_query") or "").strip()
    text_overlay = _overlay_text(opts, caption)

    _log(log_fn, f"start: mode={publish_mode} add_text={add_text} add_music={add_music} music_query_set={bool(music_query)} text_overlay_len={len(text_overlay)}")
    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    _log(log_fn, "platform.open: ok")
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})
    _log(log_fn, "platform.ensure_logged_in: ok")

    upload_result = executor.run("upload.select_file", {"platform": "tiktok", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    _log(log_fn, f"upload.select_file: payload={upload_payload}")
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"TikTok upload needs manual file selection ({reason}): {detail}")

    _log(log_fn, "form.fill(description): start")
    executor.run("form.fill", {"platform": "tiktok", "fields": {"description": caption}})
    _log(log_fn, "form.fill(description): ok")

    if add_text or add_music:
        _must_click(
            executor,
            "button[data-button-name='edit'], button.editor-entrance",
            timeout_ms=120000,
            step="open_editor",
            log_fn=log_fn,
        )

    if add_text and text_overlay:
        _must_click(executor, "div[data-name='AddTextPresetPanel']", timeout_ms=60000, step="open_text_tab", log_fn=log_fn)
        _must_click(executor, "button.AddTextPanel__addTextBasicButton", timeout_ms=60000, step="add_text_once", log_fn=log_fn)
        _must_type(executor, "textarea[name='content']", text_overlay, step="set_overlay_text", log_fn=log_fn)

    if add_music and music_query:
        _must_click(executor, "div[data-name='MusicPanel']", timeout_ms=60000, step="open_music_tab", log_fn=log_fn)
        _must_type(executor, "input[placeholder='Search sounds']", music_query, step="music_search_text", log_fn=log_fn)
        _must_click(executor, ".search-bar-container .search-icon, button[aria-label*='search' i]", timeout_ms=30000, step="music_search_submit", log_fn=log_fn)
        _log(log_fn, "music_pick_random_track: start")
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
            reason = random_track_payload.get("reason") or "unknown"
            raise RuntimeError(f"TikTok music_pick_random_track failed (reason={reason})")
        _log(log_fn, f"music_pick_random_track: ok payload={random_track_payload}")

    if add_text or (add_music and music_query):
        _must_click(
            executor,
            "button.button.Button__root--type-primary, button.Button__root--type-primary",
            timeout_ms=60000,
            step="editor_save",
            log_fn=log_fn,
        )

    _log(log_fn, f"post.submit: start mode={publish_mode}")
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
        reason = submit_payload.get("reason") or "unknown"
        raise RuntimeError(f"TikTok {publish_mode} submit failed (reason={reason})")
    _log(log_fn, f"post.submit: ok payload={submit_payload}")

    status = executor.run("post.status", {"platform": "tiktok"})
    _log(log_fn, f"post.status: {status.get('payload') or status}")
    return status
