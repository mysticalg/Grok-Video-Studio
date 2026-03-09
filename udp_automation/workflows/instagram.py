from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


_INSTAGRAM_NEXT_BUTTON_SELECTOR = (
    "div[role='button']:has-text('Next'), button:has-text('Next'), div[role='button'][tabindex='0']:has-text('Next')"
)

# User-validated CSS path for the Create menu's Post anchor. Instagram class names
# are dynamic, so this is attempted first but followed by semantic fallbacks below.
_INSTAGRAM_CREATE_MENU_POST_ANCHOR_SELECTOR = (
    "#scrollview > div > div > div.x78zum5.xdt5ytf.x1t2pt76.x1n2onr6.x1ja2u2z.x10cihs4 > "
    "div.html-div.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x9f619.x16ye13r.xvbhtw8.x78zum5.x15mokao.x1ga7v0g.x16uus16.xbiv7yw.x1uhb9sk.x1plvlek.xryxfnj.x1c4vz4f.x2lah0s.x1q0g3np.xqjyukv.x1qjc9v5.x1oa3qoh.x1qughib > "
    "div.html-div.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x9f619.xjbqb8w.x78zum5.x15mokao.x1ga7v0g.x16uus16.xbiv7yw.xixxii4.x13vifvy.x1plvlek.xryxfnj.x1c4vz4f.x2lah0s.xdt5ytf.xqjyukv.x1qjc9v5.x1oa3qoh.x1nhvcw1.x1dr59a3.xeq5yr9.x1n327nk > "
    "div > div > div > div > div > div.x6s0dn4.x78zum5.xdt5ytf.x1iyjqo2.xh8yej3 > div > div:nth-child(8) > span > div > a"
)


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

    # Instagram can keep the Create entry collapsed in the left nav until the
    # New post glyph is activated first, so prioritize that icon/link sequence.
    new_post_opened = _best_effort_click(executor, "instagram", "svg[aria-label='New post']", timeout_ms=click_timeout_ms)
    if not new_post_opened:
        new_post_opened = _best_effort_click(executor, "instagram", "a[role='link']:has(svg[aria-label='New post'])", timeout_ms=click_timeout_ms)

    # After New post is focused, the Create popover appears. First, attempt the
    # exact CSS selector captured from the failing session for the Post anchor.
    post_entry_clicked = _best_effort_click(
        executor,
        "instagram",
        _INSTAGRAM_CREATE_MENU_POST_ANCHOR_SELECTOR,
        timeout_ms=click_timeout_ms,
    )
    # If Instagram changes class names or menu structure, fall back to semantic
    # selectors that still target the first visible Post entry in the menu.
    if not post_entry_clicked:
        post_entry_clicked = _best_effort_click(
            executor,
            "instagram",
            "a[role='link']:has(span:has-text('Post'))",
            timeout_ms=click_timeout_ms,
            extra_payload={"matchIndex": 0},
        )
    if not post_entry_clicked:
        post_entry_clicked = _best_effort_click(
            executor,
            "instagram",
            "a[role='link']",
            timeout_ms=click_timeout_ms,
            extra_payload={"textContains": "post", "matchIndex": 0},
        )
    if not post_entry_clicked:
        _best_effort_click(executor, "instagram", "span:has-text('Post')", timeout_ms=click_timeout_ms)

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
