#     *    
#    * *   
#   * * *  
#  *  *  * 
# *********
#  *  *  * 
#   * * *  
#    * *   
#     *    

"""Tiktok module for Grok Video Studio automation flows."""

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



def _must_focus_any(
    executor: BaseExecutor,
    selectors: list[str],
    *,
    step: str = "",
    log_fn: LogFn | None = None,
    delay_s: float = 0.0,
    timeout_ms: int = 60000,
) -> None:
    errors: list[str] = []
    for selector in selectors:
        try:
            _must_click(
                executor,
                selector,
                timeout_ms=timeout_ms,
                step=step,
                log_fn=log_fn,
                delay_s=delay_s,
                single_click=True,
            )
            return
        except RuntimeError as exc:
            errors.append(str(exc))
            _log(log_fn, f"{step or 'focus'}: selector failed ({selector}) err={exc}")
    raise RuntimeError(f"TikTok {step or 'focus'} failed for all selectors: {' | '.join(errors)}")



def _apply_music_clip_settings(
    executor: BaseExecutor,
    *,
    music_fade_in_enabled: bool,
    music_fade_in_seconds: float,
    music_fade_out_enabled: bool,
    music_fade_out_seconds: float,
    music_volume_db: int,
    step_prefix: str,
    log_fn: LogFn | None = None,
    delay_s: float = 0.0,
) -> None:
    fade_in_input_selectors = [
        "div.PropSettingFadeInBase__wrap input.PropSettingInput__input",
        "div.PropSettingFadeInBase__fieldWrap input.PropSettingInput__input",
    ]
    fade_out_input_selectors = [
        "div.PropSettingFadeOutBase__wrap input.PropSettingInput__input",
        "div.PropSettingFadeOutBase__fieldWrap input.PropSettingInput__input",
    ]
    volume_input_selectors = [
        "div.PropSettingAudioVolume__wrap label.PropSettingInput__wrap input.PropSettingInput__input",
        "div.PropSettingAudioVolume__wrap input.PropSettingInput__input",
    ]

    if music_fade_in_enabled:
        _must_focus_any(
            executor,
            fade_in_input_selectors,
            step=f"{step_prefix}_focus_fade_in_input",
            log_fn=log_fn,
            delay_s=delay_s,
        )
        _must_type_any(
            executor,
            fade_in_input_selectors,
            f"{music_fade_in_seconds:.1f}",
            step=f"{step_prefix}_set_fade_in_seconds",
            log_fn=log_fn,
            delay_s=delay_s,
        )

    if music_fade_out_enabled:
        _must_focus_any(
            executor,
            fade_out_input_selectors,
            step=f"{step_prefix}_focus_fade_out_input",
            log_fn=log_fn,
            delay_s=delay_s,
        )
        _must_type_any(
            executor,
            fade_out_input_selectors,
            f"{music_fade_out_seconds:.1f}",
            step=f"{step_prefix}_set_fade_out_seconds",
            log_fn=log_fn,
            delay_s=delay_s,
        )

    _must_focus_any(
        executor,
        volume_input_selectors,
        step=f"{step_prefix}_focus_volume_input",
        log_fn=log_fn,
        delay_s=delay_s,
    )
    _must_type_any(
        executor,
        volume_input_selectors,
        str(music_volume_db),
        step=f"{step_prefix}_set_volume_db",
        log_fn=log_fn,
        delay_s=delay_s,
    )


def _overlay_text(opts: dict[str, Any], caption: str) -> str:
    configured = str(opts.get("text_overlay") or "").strip()
    raw = configured or str(caption or "").strip()
    without_tags = re.sub(r"(^|\s)#\w+", " ", raw)
    return " ".join(without_tags.split()).strip()


def _build_upload_filename_override(
    video_path: str,
    caption: str,
    *,
    max_chars: int = 3000,
) -> str:
    extension = Path(video_path).suffix or ".mp4"
    caption_source = " ".join(str(caption or "").split()).strip()
    if not caption_source:
        caption_source = Path(video_path).stem
    if extension and not caption_source.lower().endswith(extension.lower()):
        caption_source = f"{caption_source}{extension}"
    return caption_source[:max(1, int(max_chars))]


def run(executor: BaseExecutor, video_path: str, caption: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    log_fn = opts.get("_log_callback") if callable(opts.get("_log_callback")) else None
    publish_mode = str(opts.get("publish_mode") or "draft").strip().lower()
    if publish_mode not in {"draft", "post"}:
        publish_mode = "draft"
    add_text = bool(opts.get("add_text_overlay"))
    add_music = bool(opts.get("add_music"))
    rename_upload_filename = bool(opts.get("rename_upload_filename", True))
    upload_filename_char_limit = min(3000, max(16, int(opts.get("upload_filename_char_limit") or 167)))
    music_unique_per_add = bool(opts.get("music_unique_per_add"))
    music_add_count = max(1, min(100, int(opts.get("music_add_count") or 2)))
    music_fade_in_enabled = bool(opts.get("music_fade_in_enabled"))
    music_fade_in_seconds = min(10.0, max(0.0, round(float(opts.get("music_fade_in_seconds") or 0.0), 1)))
    music_fade_out_enabled = bool(opts.get("music_fade_out_enabled"))
    music_fade_out_seconds = min(10.0, max(0.0, round(float(opts.get("music_fade_out_seconds") or 0.0), 1)))
    music_volume_db = min(20, max(-59, int(opts.get("music_volume_db") if opts.get("music_volume_db") is not None else 0)))
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
    action_delay_ms = max(0, int(opts.get("tiktok_action_delay_ms") or opts.get("action_delay_ms") or 1000))
    action_delay_s = action_delay_ms / 1000.0
    click_timeout_ms = max(1000, int(opts.get("tiktok_click_timeout_ms") or 60000))
    editor_timeout_ms = max(click_timeout_ms, int(opts.get("tiktok_editor_timeout_ms") or 120000))
    submit_timeout_ms = max(1000, int(opts.get("tiktok_submit_timeout_ms") or 120000))

    _log(
        log_fn,
        "start: "
        f"mode={publish_mode} add_text={add_text} add_music={add_music} "
        f"rename_upload_filename={rename_upload_filename} "
        f"upload_filename_char_limit={upload_filename_char_limit} "
        f"music_add_count={music_add_count} music_unique_per_add={music_unique_per_add} "
        f"music_fade_in_enabled={music_fade_in_enabled} music_fade_in_seconds={music_fade_in_seconds:.1f} "
        f"music_fade_out_enabled={music_fade_out_enabled} music_fade_out_seconds={music_fade_out_seconds:.1f} "
        f"music_volume_db={music_volume_db} "
        f"music_queries={len(music_queries)} text_overlay_len={len(text_overlay)}",
    )
    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    _log(log_fn, "platform.open: ok")
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})
    _log(log_fn, "platform.ensure_logged_in: ok")

    upload_payload_request: dict[str, Any] = {"platform": "tiktok", "filePath": video_path}
    if rename_upload_filename and str(caption or "").strip():
        upload_file_name = _build_upload_filename_override(video_path, caption, max_chars=3000)
        upload_payload_request["fileName"] = upload_file_name
        upload_payload_request["fileNameMaxChars"] = upload_filename_char_limit
        _log(
            log_fn,
            f"upload.filename_override: chars={len(upload_file_name)} "
            "(using staged disk copy rename)",
        )
    else:
        _log(log_fn, "upload.filename_override: disabled (using original source filename)")

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
            timeout_ms=editor_timeout_ms,
            step="open_editor",
            log_fn=log_fn,
            delay_s=action_delay_s,
        )

    if add_text and text_overlay:
        _must_click(executor, "div[data-name='AddTextPresetPanel']", timeout_ms=click_timeout_ms, step="open_text_tab", log_fn=log_fn, delay_s=action_delay_s)
        _must_click(
            executor,
            "button.AddTextPanel__addTextBasicButton",
            timeout_ms=click_timeout_ms,
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
        _must_click(executor, "div[data-name='MusicPanel']", timeout_ms=click_timeout_ms, step="open_music_tab", log_fn=log_fn, delay_s=action_delay_s)
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
            timeout_ms=click_timeout_ms,
            step="music_add_track_burst",
            log_fn=log_fn,
            delay_s=action_delay_s,
            single_click=True,
            extra_payload={"burstClicks": burst_clicks},
        )

        audio_clip_selector = (
            "div.AudioClip__root.AudioClip__root--isSelected-false, "
            "div.AudioClip__root"
        )
        clips_updated: set[int] = set()

        if music_unique_per_add and len(music_queries) > 1 and music_add_count > 1:
            replacements = min(music_add_count, len(music_queries))
            _log(log_fn, f"music_replace_sequence: start replacements={replacements}")
            for replace_index in range(replacements):
                clip_number = replace_index + 1
                _must_click(
                    executor,
                    audio_clip_selector,
                    timeout_ms=click_timeout_ms,
                    step=f"music_focus_audio_clip_{clip_number}",
                    log_fn=log_fn,
                    single_click=True,
                    delay_s=action_delay_s,
                    extra_payload={"matchIndex": replace_index},
                )
                _must_type(
                    executor,
                    "input[placeholder='Search sounds']",
                    music_queries[replace_index],
                    step=f"music_search_replacement_{clip_number}",
                    log_fn=log_fn,
                    submit=True,
                    delay_s=action_delay_s,
                )
                _must_click(
                    executor,
                    "div.MusicPanelMusicItem__operation [data-testid='ArrowLeftRight'], "
                    "div.MusicPanelMusicItem__operation [data-icon='ArrowLeftRight']",
                    timeout_ms=click_timeout_ms,
                    step=f"music_replace_track_{clip_number}",
                    log_fn=log_fn,
                    single_click=True,
                    delay_s=action_delay_s,
                )
                _apply_music_clip_settings(
                    executor,
                    music_fade_in_enabled=music_fade_in_enabled,
                    music_fade_in_seconds=music_fade_in_seconds,
                    music_fade_out_enabled=music_fade_out_enabled,
                    music_fade_out_seconds=music_fade_out_seconds,
                    music_volume_db=music_volume_db,
                    step_prefix=f"music_clip_{clip_number}",
                    log_fn=log_fn,
                    delay_s=action_delay_s,
                )
                clips_updated.add(replace_index)

        for clip_index in range(burst_clicks):
            if clip_index in clips_updated:
                continue
            clip_number = clip_index + 1
            _must_click(
                executor,
                audio_clip_selector,
                timeout_ms=click_timeout_ms,
                step=f"music_focus_audio_clip_{clip_number}",
                log_fn=log_fn,
                single_click=True,
                delay_s=action_delay_s,
                extra_payload={"matchIndex": clip_index},
            )
            _apply_music_clip_settings(
                executor,
                music_fade_in_enabled=music_fade_in_enabled,
                music_fade_in_seconds=music_fade_in_seconds,
                music_fade_out_enabled=music_fade_out_enabled,
                music_fade_out_seconds=music_fade_out_seconds,
                music_volume_db=music_volume_db,
                step_prefix=f"music_clip_{clip_number}",
                log_fn=log_fn,
                delay_s=action_delay_s,
            )

    if add_text or (add_music and music_queries):
        _must_click(
            executor,
            "button.button.Button__root--type-primary .Button__content, button.button.Button__root--type-primary, button.Button__root--type-primary",
            timeout_ms=click_timeout_ms,
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
            "timeoutMs": submit_timeout_ms,
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
