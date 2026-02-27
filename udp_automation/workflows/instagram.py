from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


_INSTAGRAM_NEXT_BUTTON_SELECTOR = (
    "div[role='button']:has-text('Next'), button:has-text('Next'), div[role='button'][tabindex='0']:has-text('Next')"
)

_INSTAGRAM_CREATE_LINK_SELECTOR = (
    "a[role='link'][href='#'].x1i10hfl.x1qjc9v5.xjbqb8w.xjqpnuy.xc5r6h4.xqeqjp1.x1phubyo.x13fuv20"
)

_INSTAGRAM_CREATE_LINK_SELECTOR_ALT = (
    "a[role='link'][href='#'].x1i10hfl.xjbqb8w.x1ejq31n.x18oe1m7.x1sy0etr.xstzfhl"
)


def _best_effort_click(executor: BaseExecutor, platform: str, selector: str, timeout_ms: int = 8000) -> None:
    try:
        executor.run("dom.click", {"platform": platform, "selector": selector, "timeoutMs": timeout_ms})
    except Exception:
        return




def _best_effort_click_wrapped_text(
    executor: BaseExecutor,
    platform: str,
    text_value: str,
    closest_selector: str = "a[role='link'], a, div[role='button'], button",
    timeout_ms: int = 8000,
) -> None:
    try:
        executor.run(
            "dom.click",
            {
                "platform": platform,
                "text": text_value,
                "closestSelector": closest_selector,
                "timeoutMs": timeout_ms,
            },
        )
    except Exception:
        return

def _safe_instagram_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if candidate.lower().startswith(("http://", "https://")):
        return candidate
    return "https://www.instagram.com/"


def run(executor: BaseExecutor, video_path: str, caption: str, platform_url: str = "") -> dict[str, Any]:
    executor.run(
        "platform.open",
        {
            "platform": "instagram",
            "url": _safe_instagram_url(platform_url),
            "reuseTab": True,
        },
    )
    executor.run("platform.ensure_logged_in", {"platform": "instagram"})

    # Start from profile/home page: hover the collapsed sidebar rail so Create is revealed,
    # then open Create -> Post and attach media.
    executor.run(
        "dom.hover",
        {
            "platform": "instagram",
            "selector": "div.x1cy8zhl.x9f619.x78zum5.xdt5ytf.x1gvbg2u.x1qughib[style*='width: 72px']",
            "timeoutMs": 10000,
        },
    )
    # Prefer wrapped text targeting (find <span>Create</span>/<span>Post</span> and click the wrapping link/button).
    _best_effort_click_wrapped_text(executor, "instagram", "Create", timeout_ms=12000)
    _best_effort_click(executor, "instagram", _INSTAGRAM_CREATE_LINK_SELECTOR, timeout_ms=12000)
    _best_effort_click(executor, "instagram", _INSTAGRAM_CREATE_LINK_SELECTOR_ALT, timeout_ms=12000)
    _best_effort_click(executor, "instagram", "a[href*='create']", timeout_ms=10000)

    _best_effort_click_wrapped_text(executor, "instagram", "Post", closest_selector="div", timeout_ms=12000)
    _best_effort_click(executor, "instagram", "a[href*='create']", timeout_ms=10000)

    _best_effort_click(executor, "instagram", "button[aria-label*='Select from computer' i]")
    _best_effort_click(executor, "instagram", "div[role='button'][aria-label*='Select from computer' i]")

    upload_result = executor.run("upload.select_file", {"platform": "instagram", "filePath": video_path})
    upload_payload = upload_result.get("payload") or {}
    if upload_payload.get("requiresUserAction"):
        reason = upload_payload.get("reason") or "manual_file_selection_required"
        detail = upload_payload.get("message") or "automatic file input was not found"
        raise RuntimeError(f"Instagram upload needs manual file selection ({reason}): {detail}")

    # Instagram post flow requires two Next clicks before caption/share.
    _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=12000)
    _best_effort_click(executor, "instagram", _INSTAGRAM_NEXT_BUTTON_SELECTOR, timeout_ms=12000)

    executor.run("form.fill", {"platform": "instagram", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "instagram"})
    return executor.run("post.status", {"platform": "instagram"})
