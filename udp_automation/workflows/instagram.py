from __future__ import annotations

import time
from typing import Any

from udp_automation.executors import BaseExecutor


_INSTAGRAM_NEXT_BUTTON_SELECTOR = (
    "div[role='button']:has-text('Next'), button:has-text('Next'), div[role='button'][tabindex='0']:has-text('Next')"
)


def _is_user_stop_error(exc: Exception) -> bool:
    """Return True when the automation engine reports an explicit user stop."""
    return "stopped by user" in str(exc).lower()


def _pause_between_actions(delay_ms: int) -> None:
    """Keep interactions stable by sleeping briefly between high-level actions."""
    if delay_ms <= 0:
        return
    time.sleep(delay_ms / 1000.0)


def _require_step(step_name: str, completed: bool) -> None:
    """Fail fast when a required Instagram automation step does not complete."""
    if completed:
        return
    raise RuntimeError(f"Instagram workflow required step failed: {step_name}")


def _best_effort_click(
    executor: BaseExecutor,
    platform: str,
    selector: str,
    timeout_ms: int = 8000,
    extra_payload: dict[str, Any] | None = None,
) -> bool:
    log_callback = getattr(executor, "log_callback", None)
    if callable(log_callback):
        try:
            log_callback(
                f"Instagram workflow: attempting dom.click target={{platform={platform}, selector={selector!r}}} timeout_ms={timeout_ms} retry_mode=best_effort"
            )
        except Exception:
            pass
    try:
        payload = {"platform": platform, "selector": selector, "timeoutMs": timeout_ms}
        if extra_payload:
            payload.update(extra_payload)
        response = executor.run("dom.click", payload)
        if callable(log_callback):
            try:
                response_payload = response.get("payload") or {}
                log_callback(
                    "Instagram workflow: dom.click completed "
                    f"target={{platform={platform}, selector={selector!r}}} clicked={bool(response_payload.get('clicked'))}"
                )
            except Exception:
                pass
        return bool((response.get("payload") or {}).get("clicked"))
    except Exception as exc:
        if callable(log_callback):
            try:
                log_callback(
                    "Instagram workflow: dom.click best-effort step failed "
                    f"target={{platform={platform}, selector={selector!r}}} timeout_ms={timeout_ms} error={exc}"
                )
            except Exception:
                pass
        if _is_user_stop_error(exc):
            raise
        return False



def _safe_instagram_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if candidate.lower().startswith(("http://", "https://")):
        return candidate
    return "https://www.instagram.com/"


def run(executor: BaseExecutor, video_path: str, caption: str, platform_url: str = "", options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    click_timeout_ms = max(1000, int(opts.get("instagram_click_timeout_ms") or 10000))
    next_timeout_ms = max(1000, int(opts.get("instagram_next_timeout_ms") or 12000))
    # Small per-action pause keeps Instagram menu transitions stable.
    action_delay_ms = max(0, int(opts.get("instagram_action_delay_ms") or 350))
    log_callback = getattr(executor, "log_callback", None)
    if callable(log_callback):
        try:
            log_callback(
                "Instagram workflow: starting run "
                f"url={_safe_instagram_url(platform_url)!r} video_path={video_path!r} caption_chars={len(caption or '')}"
            )
        except Exception:
            pass

    executor.run(
        "platform.open",
        {
            "platform": "instagram",
            "url": _safe_instagram_url(platform_url),
            "reuseTab": True,
        },
    )
    _pause_between_actions(action_delay_ms)
    executor.run("platform.ensure_logged_in", {"platform": "instagram"})
    _pause_between_actions(action_delay_ms)

    # Prefer the Create trigger first to align with current Instagram nav flow.
    create_menu_opened = _best_effort_click(
        executor,
        "instagram",
        "a[role='link']",
        timeout_ms=click_timeout_ms,
        extra_payload={"textContains": "create", "matchIndex": 0},
    )
    _pause_between_actions(action_delay_ms)
    # Fallback: some layouts expose New post glyph instead of visible Create text.
    if not create_menu_opened:
        create_menu_opened = _best_effort_click(
            executor,
            "instagram",
            "svg[aria-label='New post']",
            timeout_ms=click_timeout_ms,
        )
        _pause_between_actions(action_delay_ms)
    if not create_menu_opened:
        create_menu_opened = _best_effort_click(
            executor,
            "instagram",
            "a[role='link']:has(svg[aria-label='New post'])",
            timeout_ms=click_timeout_ms,
        )
        _pause_between_actions(action_delay_ms)
    _require_step("open_create_menu", create_menu_opened)

    # Click the Post entry directly after opening the Create menu.
    post_entry_clicked = _best_effort_click(
        executor,
        "instagram",
        "a[role='link']",
        timeout_ms=click_timeout_ms,
        extra_payload={"textContains": "post", "matchIndex": 0},
    )
    _pause_between_actions(action_delay_ms)

    # Keep a light text fallback if Instagram changes the nested link markup.
    if not post_entry_clicked:
        post_entry_clicked = _best_effort_click(
            executor,
            "instagram",
            "span",
            timeout_ms=click_timeout_ms,
            extra_payload={"textContains": "post", "matchIndex": 0},
        )
        _pause_between_actions(action_delay_ms)
    _require_step("select_post_entry", post_entry_clicked)

    select_from_computer_clicked = _best_effort_click(
        executor,
        "instagram",
        "button[aria-label*='Select from computer' i]",
        timeout_ms=click_timeout_ms,
    )
    _pause_between_actions(action_delay_ms)
    if not select_from_computer_clicked:
        select_from_computer_clicked = _best_effort_click(
            executor,
            "instagram",
            "div[role='button'][aria-label*='Select from computer' i]",
            timeout_ms=click_timeout_ms,
        )
    _pause_between_actions(action_delay_ms)
    _require_step("open_select_from_computer", select_from_computer_clicked)

    upload_result = executor.run("upload.select_file", {"platform": "instagram", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"Instagram upload needs manual file selection ({reason}): {detail}")

    _pause_between_actions(action_delay_ms)
    next_first_clicked = _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=next_timeout_ms)
    _pause_between_actions(action_delay_ms)
    _require_step("next_button_first_click", next_first_clicked)
    next_second_clicked = _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=next_timeout_ms)
    _pause_between_actions(action_delay_ms)
    _require_step("next_button_second_click", next_second_clicked)

    executor.run("form.fill", {"platform": "instagram", "fields": {"description": caption}})
    _pause_between_actions(action_delay_ms)
    executor.run("post.submit", {"platform": "instagram"})
    _pause_between_actions(action_delay_ms)
    status = executor.run("post.status", {"platform": "instagram"})
    if callable(log_callback):
        try:
            log_callback(
                "Instagram workflow: finished run "
                f"post_status_payload={status.get('payload') or {}}"
            )
        except Exception:
            pass
    return status
