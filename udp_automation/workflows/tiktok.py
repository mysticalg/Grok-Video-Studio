from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from udp_automation.executors import BaseExecutor


LogFn = Callable[[str], None]


def _log(log_fn: LogFn | None, message: str) -> None:
    if callable(log_fn):
        try:
            log_fn(message)
        except Exception:
            pass


def _pause(delay_s: float, *, step: str = "", log_fn: LogFn | None = None) -> None:
    if delay_s <= 0:
        return
    _log(log_fn, f"{step or 'pause'}: sleeping {delay_s:.1f}s")
    time.sleep(delay_s)


def _must_click(
    executor: BaseExecutor,
    selector: str,
    timeout_ms: int = 30000,
    *,
    step: str = "",
    log_fn: LogFn | None = None,
    text_contains: str = "",
    delay_s: float = 0.0,
    single_click: bool = False,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _log(log_fn, f"{step or 'click'}: selector={selector} timeout_ms={timeout_ms}")
    click_payload: dict[str, Any] = {
        "platform": "tiktok",
        "selector": selector,
        "timeoutMs": timeout_ms,
        "textContains": text_contains,
        "singleClick": single_click,
    }
    if isinstance(extra_payload, dict) and extra_payload:
        click_payload.update(extra_payload)
    result = executor.run("dom.click", click_payload)
    payload = result.get("payload") or {}
    if payload.get("clicked") is False:
        reason = payload.get("reason") or "unknown"
        raise RuntimeError(f"TikTok {step or 'click'} failed (selector={selector}, reason={reason})")
    _log(log_fn, f"{step or 'click'}: ok payload={payload}")
    _pause(delay_s, step=f"{step or 'click'}_settle", log_fn=log_fn)
    return payload


def _must_type(
    executor: BaseExecutor,
    selector: str,
    value: str,
    *,
    step: str = "",
    log_fn: LogFn | None = None,
    submit: bool = False,
    delay_s: float = 0.0,
) -> None:
    _log(log_fn, f"{step or 'type'}: selector={selector} chars={len(value)}")
    result = executor.run(
        "dom.type",
        {
            "platform": "tiktok",
            "selector": selector,
            "value": value,
            "submit": submit,
        },
    )
    payload = result.get("payload") or {}
    if payload.get("typed") is False:
        raise RuntimeError(f"TikTok {step or 'type'} failed (selector={selector})")
    _log(log_fn, f"{step or 'type'}: ok payload={payload}")
    _pause(delay_s, step=f"{step or 'type'}_settle", log_fn=log_fn)


def _must_type_any(
    executor: BaseExecutor,
    selectors: list[str],
    value: str,
    *,
    step: str = "",
    log_fn: LogFn | None = None,
    submit: bool = False,
    delay_s: float = 0.0,
) -> None:
    errors: list[str] = []
    for selector in selectors:
        try:
            _must_type(
                executor,
                selector,
                value,
                step=step,
                log_fn=log_fn,
                submit=submit,
                delay_s=delay_s,
            )
            return
        except RuntimeError as exc:
            errors.append(str(exc))
            _log(log_fn, f"{step or 'type'}: selector failed ({selector}) err={exc}")
    raise RuntimeError(f"TikTok {step or 'type'} failed for all selectors: {' | '.join(errors)}")


def _overlay_text(opts: dict[str, Any], caption: str) -> str:
    configured = str(opts.get("text_overlay") or "").strip()
    raw = configured or str(caption or "").strip()
    without_tags = re.sub(r"(^|\s)#\w+", " ", raw)
    return " ".join(without_tags.split()).strip()


def _sanitize_filename_stem(name: str) -> str:
    # Keep filename safe across platforms and strip all dots from the stem.
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", str(name or ""))
    cleaned = cleaned.replace(".", " ")
    collapsed = " ".join(cleaned.split()).strip(" .")
    return collapsed or "tiktok_upload"


def _build_upload_filename(
    video_path: str,
    caption: str,
    *,
    max_caption_chars: int = 3000,
    max_filename_chars: int = 1000,
) -> str:
    extension = Path(video_path).suffix or ".mp4"
    caption_source = str(caption or "").strip()[:max_caption_chars]
    safe_stem = _sanitize_filename_stem(caption_source)
    stem_limit = max(1, max_filename_chars - len(extension))
    stem = (safe_stem[:stem_limit].rstrip(" _-") or "tiktok_upload")
    return f"{stem}{extension}"


def run(executor: BaseExecutor, video_path: str, caption: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    log_fn = opts.get("_log_callback") if callable(opts.get("_log_callback")) else None
    publish_mode = str(opts.get("publish_mode") or "draft").strip().lower()
    if publish_mode not in {"draft", "post"}:
        publish_mode = "draft"
    add_text = bool(opts.get("add_text_overlay"))
    add_music = bool(opts.get("add_music"))
    music_unique_per_add = bool(opts.get("music_unique_per_add"))
    music_add_count = max(1, min(10, int(opts.get("music_add_count") or 2)))
    raw_music_queries = opts.get("music_queries_effective")
    music_queries: list[str] = []
    if isinstance(raw_music_queries, list):
        music_queries = [str(item).strip() for item in raw_music_queries if str(item or "").strip()]
    fallback_query = str(opts.get("music_query_effective") or opts.get("music_query") or "").strip()
    if not music_queries and fallback_query:
        music_queries = [fallback_query]

    if music_queries:
        if len(music_queries) < music_add_count:
            last_query = music_queries[-1]
            music_queries.extend([last_query] * (music_add_count - len(music_queries)))
        elif len(music_queries) > music_add_count:
            music_queries = music_queries[:music_add_count]
    text_overlay = _overlay_text(opts, caption)
    action_delay_s = float(opts.get("action_delay_s") or 2.0)
    startup_delay_s = float(opts.get("startup_delay_s") or action_delay_s)

    _log(
        log_fn,
        "start: "
        f"mode={publish_mode} add_text={add_text} add_music={add_music} "
        f"music_add_count={music_add_count} music_unique_per_add={music_unique_per_add} "
        f"music_queries={len(music_queries)} text_overlay_len={len(text_overlay)}",
    )
    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    _log(log_fn, "platform.open: ok")
    _pause(startup_delay_s, step="startup_wait_after_open", log_fn=log_fn)
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})
    _log(log_fn, "platform.ensure_logged_in: ok")
    _pause(action_delay_s, step="startup_wait_after_login_check", log_fn=log_fn)

    upload_payload_request: dict[str, Any] = {"platform": "tiktok", "filePath": video_path}
    if str(caption or "").strip():
        upload_file_name = _build_upload_filename(video_path, caption, max_caption_chars=3000)
        upload_payload_request["fileName"] = upload_file_name
        _log(log_fn, f"upload.filename_override: fileName={upload_file_name} caption_chars={len(str(caption or '').strip())}")

    upload_result = executor.run("upload.select_file", upload_payload_request)
    upload_payload = upload_result.get("payload") or {}
    _log(log_fn, f"upload.select_file: payload={upload_payload}")
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"TikTok upload needs manual file selection ({reason}): {detail}")
    _pause(action_delay_s, step="wait_after_upload_select", log_fn=log_fn)

    if add_text or add_music:
        _must_click(
            executor,
            "button[data-button-name='edit'], button.editor-entrance",
            timeout_ms=120000,
            step="open_editor",
            log_fn=log_fn,
            delay_s=action_delay_s,
        )

    if add_text and text_overlay:
        _must_click(executor, "div[data-name='AddTextPresetPanel']", timeout_ms=60000, step="open_text_tab", log_fn=log_fn, delay_s=action_delay_s)
        _must_click(
            executor,
            "button.AddTextPanel__addTextBasicButton",
            timeout_ms=60000,
            step="add_text_once",
            log_fn=log_fn,
            delay_s=action_delay_s,
            single_click=True,
        )
        _must_type_any(
            executor,
            ["textarea[name='content']:focus", "textarea[name='content']"],
            text_overlay,
            step="set_overlay_text",
            log_fn=log_fn,
            delay_s=action_delay_s,
        )

    if add_music and music_queries:
        _must_click(executor, "div[data-name='MusicPanel']", timeout_ms=60000, step="open_music_tab", log_fn=log_fn, delay_s=action_delay_s)
        if len(music_queries) > 1 and music_add_count > 1:
            _log(
                log_fn,
                "music_add_burst: multiple queries were provided, but multi-add now uses the first query and bursts Add on one selected sound",
            )
        _must_type(
            executor,
            "input[placeholder='Search sounds']",
            music_queries[0],
            step="music_search_text_1",
            log_fn=log_fn,
            submit=True,
            delay_s=action_delay_s,
        )
        burst_clicks = music_add_count if music_add_count > 1 else 1
        _log(log_fn, f"music_add_burst: triggering add button {burst_clicks} time(s) in one action")
        _must_click(
            executor,
            "div.MusicPanelMusicItem__operation button[role='button'], div.MusicPanelMusicItem__operation button",
            timeout_ms=60000,
            step="music_add_track_burst",
            log_fn=log_fn,
            delay_s=action_delay_s,
            single_click=True,
            extra_payload={"burstClicks": burst_clicks},
        )

        if music_unique_per_add and len(music_queries) > 1 and music_add_count > 1:
            replacements = min(music_add_count, len(music_queries))
            _log(log_fn, f"music_replace_sequence: start replacements={replacements}")
            for replace_index in range(replacements):
                _must_click(
                    executor,
                    "div.AudioClip__root.AudioClip__root--isSelected-false, "
                    "div.AudioClip__root",
                    timeout_ms=60000,
                    step=f"music_focus_audio_clip_{replace_index + 1}",
                    log_fn=log_fn,
                    single_click=True,
                    delay_s=action_delay_s,
                    extra_payload={"matchIndex": replace_index},
                )
                _must_type(
                    executor,
                    "input[placeholder='Search sounds']",
                    music_queries[replace_index],
                    step=f"music_search_replacement_{replace_index + 1}",
                    log_fn=log_fn,
                    submit=True,
                    delay_s=action_delay_s,
                )
                _must_click(
                    executor,
                    "div.MusicPanelMusicItem__operation [data-testid='ArrowLeftRight'], "
                    "div.MusicPanelMusicItem__operation [data-icon='ArrowLeftRight']",
                    timeout_ms=60000,
                    step=f"music_replace_track_{replace_index + 1}",
                    log_fn=log_fn,
                    single_click=True,
                    delay_s=action_delay_s,
                )

    if add_text or (add_music and music_queries):
        _must_click(
            executor,
            "button.button.Button__root--type-primary .Button__content, button.button.Button__root--type-primary, button.Button__root--type-primary",
            timeout_ms=60000,
            step="editor_save",
            log_fn=log_fn,
            text_contains="save",
            delay_s=action_delay_s,
        )

    _log(log_fn, f"post.submit: start mode={publish_mode}")
    submit_result = executor.run(
        "post.submit",
        {
            "platform": "tiktok",
            "mode": publish_mode,
            "waitForUpload": True,
            "timeoutMs": 120000,
            "singleClick": True,
        },
    )
    submit_payload = submit_result.get("payload") or {}
    if submit_payload and submit_payload.get("clicked") is False:
        reason = submit_payload.get("reason") or "unknown"
        raise RuntimeError(f"TikTok {publish_mode} submit failed (reason={reason})")
    _log(log_fn, f"post.submit: ok payload={submit_payload}")
    _pause(action_delay_s, step="wait_after_submit", log_fn=log_fn)

    status = executor.run("post.status", {"platform": "tiktok"})
    _log(log_fn, f"post.status: {status.get('payload') or status}")
    return status
