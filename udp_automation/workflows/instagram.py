#     *    
#    * *   
#   * * *  
#  *  *  * 
# *********
#  *  *  * 
#   * * *  
#    * *   
#     *    

"""Instagram module for Grok Video Studio automation flows."""

from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


_INSTAGRAM_NEXT_BUTTON_SELECTOR = (
    "div[role='button']:has-text('Next'), button:has-text('Next'), div[role='button'][tabindex='0']:has-text('Next')"
)


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str, timeout_ms: int = 8000) -> None:
    log_callback = getattr(executor, "log_callback", None)
    if callable(log_callback):
        try:
            log_callback(
                f"Instagram workflow: attempting dom.click target={{platform={platform}, selector={selector!r}}} timeout_ms={timeout_ms} retry_mode=best_effort"
            )
        except Exception:
            pass
    try:
        response = executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": timeout_ms})
        if callable(log_callback):
            try:
                payload = response.get("payload") or {}
                log_callback(
                    "Instagram workflow: dom.click completed "
                    f"target={{platform={platform}, selector={selector!r}}} clicked={bool(payload.get('clicked'))}"
                )
            except Exception:
                pass
    except Exception as exc:
        if callable(log_callback):
            try:
                log_callback(
                    "Instagram workflow: dom.click best-effort step failed "
                    f"target={{platform={platform}, selector={selector!r}}} timeout_ms={timeout_ms} error={exc}"
                )
            except Exception:
                pass
        return


def _safe_instagram_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if candidate.lower().startswith(("http://", "https://")):
        return candidate
    return "https://www.instagram.com/"


def run(executor: BaseExecutor, video_path: str, caption: str, platform_url: str = "", options: dict[str, Any] | None = None) -> dict[str, Any]:
    opts = options or {}
    click_timeout_ms = max(1000, int(opts.get("instagram_click_timeout_ms") or 10000))
    next_timeout_ms = max(1000, int(opts.get("instagram_next_timeout_ms") or 12000))
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
    executor.run("platform.ensure_logged_in", {"platform": "instagram"})

    _best_effort_click(executor, "instagram", "span:has-text('Create')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "div[role='button']:has-text('Create')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "a[href*='create']", timeout_ms=click_timeout_ms)
    # Instagram can keep the "Create" text hidden until the side-nav "New post"
    # glyph is focused/hovered. Click the glyph first so the label/menu is revealed.
    _best_effort_click(executor, "instagram", "svg[aria-label='New post']", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "title:has-text('New post')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "a[role='link']:has(svg[aria-label='New post'])", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "span:has-text('Create')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "div[role='button']:has-text('Create')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "span:has-text('Post')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "div[role='button']:has-text('Post')", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "a[href*='create']", timeout_ms=click_timeout_ms)

    _best_effort_click(executor, "instagram", "button[aria-label*='Select from computer' i]", timeout_ms=click_timeout_ms)
    _best_effort_click(executor, "instagram", "div[role='button'][aria-label*='Select from computer' i]", timeout_ms=click_timeout_ms)

    upload_result = executor.run("upload.select_file", {"platform": "instagram", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"Instagram upload needs manual file selection ({reason}): {detail}")

    _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=next_timeout_ms)
    _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=next_timeout_ms)

    executor.run("form.fill", {"platform": "instagram", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "instagram"})
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
