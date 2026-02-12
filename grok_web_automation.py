from __future__ import annotations

import os
import time
from pathlib import Path


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Playwright is required for web automation. Install dependencies and run 'playwright install chromium'."
        ) from exc
    return sync_playwright


def _get_selectors() -> dict[str, str]:
    return {
        "imagine_url": os.getenv("GROK_IMAGINE_URL", "https://grok.com/imagine"),
        "prompt": os.getenv("GROK_IMAGINE_PROMPT_SELECTOR", "textarea"),
        "submit": os.getenv("GROK_IMAGINE_SUBMIT_SELECTOR", "button:has-text('Generate')"),
        "video": os.getenv("GROK_IMAGINE_VIDEO_SELECTOR", "video"),
    }


def manual_login_and_save(storage_state_path: Path, timeout_s: int = 300) -> None:
    sync_playwright = _require_playwright()
    selectors = _get_selectors()
    storage_state_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(selectors["imagine_url"], wait_until="domcontentloaded")
        page.wait_for_selector(selectors["prompt"], timeout=timeout_s * 1000)
        context.storage_state(path=str(storage_state_path))
        browser.close()


def generate_video_via_web(
    storage_state_path: Path,
    prompt: str,
    output_path: Path,
    timeout_s: int = 360,
) -> Path:
    sync_playwright = _require_playwright()
    selectors = _get_selectors()

    if not storage_state_path.exists():
        raise RuntimeError("No saved web login session found. Run manual web login first.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(storage_state_path))
        page = context.new_page()
        page.goto(selectors["imagine_url"], wait_until="domcontentloaded")

        page.fill(selectors["prompt"], prompt)
        page.click(selectors["submit"])

        page.wait_for_selector(selectors["video"], timeout=timeout_s * 1000)
        video_el = page.locator(selectors["video"]).first
        video_url = video_el.get_attribute("src")
        if not video_url:
            # try source child
            video_url = page.locator(f"{selectors['video']} source").first.get_attribute("src")
        if not video_url:
            raise RuntimeError("Generated video element found but no source URL was discovered.")

        response = context.request.get(video_url)
        if not response.ok:
            raise RuntimeError(f"Could not download generated video from web session: {response.status}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.body())
        browser.close()

    return output_path
