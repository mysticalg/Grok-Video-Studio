from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.cdp_controller import CDPController
from automation.chrome_manager import AutomationChromeManager


GROK_IMAGINE_URL = "https://grok.com/imagine"
POST_RE = re.compile(r"/imagine/post/([0-9a-fA-F-]{20,})")


def _default_output_dir(root: Path) -> Path:
    return root / "downloads" / f"cdp-grok-smoke-{time.strftime('%Y%m%d-%H%M%S')}"


async def _remove_attached_images(page) -> int:
    removed = 0
    for _ in range(4):
        result = await page.evaluate(
            r"""
            (() => {
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const button = [...document.querySelectorAll("button,[role='button']")]
                    .find((el) => isVisible(el) && /remove\s+image/i.test(`${el.getAttribute("aria-label") || ""} ${el.textContent || ""}`));
                if (!button) return false;
                try { button.click(); return true; } catch (_) {}
                try { button.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true })); return true; } catch (_) {}
                return false;
            })()
            """
        )
        if not result:
            break
        removed += 1
        await page.wait_for_timeout(500)
    return removed


async def _attached_image_state(page) -> dict[str, Any]:
    return await page.evaluate(
        r"""
        (() => {
            const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
            const removeImageVisible = !![...document.querySelectorAll("button,[role='button']")]
                .find((el) => isVisible(el) && /remove\s+image/i.test(`${el.getAttribute("aria-label") || ""} ${el.textContent || ""}`));
            const fileInputs = [...document.querySelectorAll("input[type='file']")]
                .map((el) => ({ accept: el.getAttribute("accept") || "", files: el.files ? el.files.length : 0 }));
            return { attachedImage: removeImageVisible, removeImageVisible, fileInputs, href: location.href };
        })()
        """
    )


async def _wait_for_attachment(page, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = await _attached_image_state(page)
        if last.get("attachedImage"):
            return last
        await page.wait_for_timeout(500)
    return last


async def _upload_image(page, cdp: CDPController, image_path: Path) -> dict[str, Any]:
    await page.evaluate(
        r"""
        (() => {
            const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
            const prompt = [...document.querySelectorAll("div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'], textarea, input")]
                .find((el) => isVisible(el));
            if (!prompt) return false;
            try { prompt.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
            try { prompt.focus({ preventScroll: true }); } catch (_) { try { prompt.focus(); } catch (_) {} }
            return true;
        })()
        """
    )

    selectors = [
        "input[type='file'][accept*='image']",
        "input[type='file'][accept='image/*']",
        "form input[type='file'][name='files']",
        "input[type='file']",
    ]
    errors: list[str] = []
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = await locator.count()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{selector}: {exc}")
            continue
        for index in range(min(count, 8)):
            try:
                await locator.nth(index).set_input_files(str(image_path), timeout=10_000)
                state = await _wait_for_attachment(page)
                return {
                    "ok": bool(state.get("attachedImage")),
                    "mode": "playwright_set_input_files",
                    "selector": selector,
                    "index": index,
                    "state": state,
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{selector}[{index}]: {exc}")

    for selector in selectors:
        try:
            if await cdp.set_file_input_files_via_dom(page, selector, str(image_path)):
                state = await _wait_for_attachment(page)
                return {
                    "ok": bool(state.get("attachedImage")),
                    "mode": "cdp_dom_set_file_input_files",
                    "selector": selector,
                    "index": 0,
                    "state": state,
                }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{selector} CDP: {exc}")

    return {"ok": False, "mode": "failed", "errors": errors[-8:], "state": await _attached_image_state(page)}


async def _submit_video_prompt(page, prompt: str, quality: str, duration: str) -> dict[str, Any]:
    return await page.evaluate(
        r"""
        async ({ prompt, quality, duration }) => {
            const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
            const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
            const textOf = (el) => String(el?.textContent || "").replace(/\s+/g, " ").trim();
            const desc = (el) => `${el?.getAttribute?.("aria-label") || ""} ${el?.getAttribute?.("title") || ""} ${textOf(el)}`.trim();
            const click = (el) => {
                if (!el || !isVisible(el) || el.disabled) return false;
                try { el.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
                try { el.focus({ preventScroll: true }); } catch (_) {}
                const common = { bubbles: true, cancelable: true, composed: true };
                for (const pair of [["pointerdown", PointerEvent], ["mousedown", MouseEvent], ["pointerup", PointerEvent], ["mouseup", MouseEvent], ["click", MouseEvent]]) {
                    try { el.dispatchEvent(new pair[1](pair[0], common)); } catch (_) {}
                }
                try { el.click(); } catch (_) {}
                return true;
            };
            const clickExact = (label) => {
                const pattern = new RegExp(`^\\s*${String(label).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*$`, "i");
                const button = [...document.querySelectorAll("button,[role='button']")]
                    .filter(isVisible)
                    .find((el) => pattern.test(desc(el)));
                return button ? click(button) : false;
            };

            clickExact("Video");
            await sleep(250);
            clickExact(quality);
            await sleep(250);
            clickExact(duration);
            await sleep(250);

            const promptSelectors = [
                "div.tiptap.ProseMirror[contenteditable='true']",
                "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                "[contenteditable='true'][aria-label*='Type to imagine' i]",
                "[contenteditable='true']",
                "textarea[placeholder*='Type to imagine' i]",
                "input[placeholder*='Type to imagine' i]"
            ];
            const promptInput = promptSelectors
                .flatMap((selector) => [...document.querySelectorAll(selector)])
                .find(isVisible) || null;
            if (!promptInput) return { ok: false, stage: "prompt-not-found", href: location.href };

            try { promptInput.scrollIntoView({ block: "center" }); promptInput.focus(); } catch (_) {}
            if (promptInput.isContentEditable) {
                const paragraph = document.createElement("p");
                paragraph.textContent = prompt;
                promptInput.replaceChildren(paragraph);
            } else {
                const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(promptInput), "value")?.set;
                if (setter) setter.call(promptInput, prompt);
                else promptInput.value = prompt;
            }
            promptInput.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: prompt, inputType: "insertText" }));
            promptInput.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
            await sleep(400);
            const filled = promptInput.isContentEditable ? textOf(promptInput) : String(promptInput.value || "").trim();
            if (!filled) return { ok: false, stage: "prompt-empty", href: location.href };

            const submit = [...document.querySelectorAll("button[type='submit'][aria-label='Submit'], button[aria-label='Submit'], button[type='submit']")]
                .find((el) => isVisible(el) && !el.disabled) || null;
            if (!submit) return { ok: false, stage: "submit-not-found", filledLength: filled.length, href: location.href };
            click(submit);
            return { ok: true, stage: "submitted", filledLength: filled.length, href: location.href };
        }
        """,
        {"prompt": prompt, "quality": quality, "duration": duration},
    )


async def _generation_state(page) -> dict[str, Any]:
    return await page.evaluate(
        r"""
        (() => {
            const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
            const text = document.body ? String(document.body.innerText || "") : "";
            const href = String(location.href || "");
            const videos = [...document.querySelectorAll("video, video source")]
                .map((el) => {
                    const host = el.tagName && el.tagName.toLowerCase() === "source" ? el.closest("video") : el;
                    return String(el.currentSrc || el.src || host?.currentSrc || host?.src || "").trim();
                })
                .filter(Boolean);
            const downloadVisible = [...document.querySelectorAll("button,[role='button'],a[download]")]
                .some((el) => isVisible(el) && !el.disabled && /download/i.test(`${el.getAttribute("aria-label") || ""} ${el.textContent || ""}`));
            const progressMatches = [...text.matchAll(/(\d{1,3})\s*%/g)]
                .map((match) => Number(match[1]))
                .filter((value) => Number.isFinite(value));
            const progress = progressMatches.length ? Math.max(...progressMatches) : null;
            const generating = /generating|rendering|cancel\s+video/i.test(text);
            const blocked = /unable|failed|moderation|policy|try again|something went wrong/i.test(text);
            return { href, videos, downloadVisible, progress, generating, blocked, tail: text.slice(-700) };
        })()
        """
    )


async def _wait_for_video_ready(page, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_log = 0.0
    last_state: dict[str, Any] = {}
    while time.time() < deadline:
        await page.wait_for_timeout(3000)
        last_state = await _generation_state(page)
        if last_state.get("downloadVisible") or last_state.get("videos"):
            return last_state
        now = time.time()
        if now - last_log >= 20:
            print(
                "POLL="
                + json.dumps(
                    {key: last_state.get(key) for key in ("href", "progress", "generating", "downloadVisible", "blocked")},
                    ensure_ascii=True,
                ),
                flush=True,
            )
            last_log = now
        if last_state.get("blocked") and not last_state.get("generating"):
            raise RuntimeError("Grok reported a failure/block state: " + json.dumps(last_state, ensure_ascii=True)[:1000])
    raise TimeoutError("Timed out waiting for Grok video readiness. Last state: " + json.dumps(last_state, ensure_ascii=True)[:1000])


async def _download_video(page, state: dict[str, Any], output_path: Path) -> dict[str, Any]:
    direct_url = ""
    for candidate in state.get("videos") or []:
        if isinstance(candidate, str) and candidate.startswith("http"):
            direct_url = candidate
            break

    if direct_url:
        response = await page.request.get(direct_url, timeout=180_000)
        if not response.ok:
            raise RuntimeError(f"Direct video download failed with HTTP {response.status}")
        output_path.write_bytes(await response.body())
        return {"mode": "direct-video-src", "directUrl": direct_url}

    async with page.expect_download(timeout=180_000) as download_info:
        clicked = await page.evaluate(
            r"""
            (() => {
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const button = [...document.querySelectorAll("button,[role='button'],a[download]")]
                    .find((el) => isVisible(el) && !el.disabled && /download/i.test(`${el.getAttribute("aria-label") || ""} ${el.textContent || ""}`));
                if (!button) return false;
                try { button.scrollIntoView({ block: "center", inline: "center" }); } catch (_) {}
                try { button.click(); return true; } catch (_) {}
                try { button.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, composed: true })); return true; } catch (_) {}
                return false;
            })()
            """
        )
        if not clicked:
            raise RuntimeError("Download button disappeared before click")
    download = await download_info.value
    await download.save_as(str(output_path))
    return {"mode": "browser-download-button", "suggestedFilename": download.suggested_filename}


def _valid_mp4(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 250_000:
        return False
    header = path.read_bytes()[:12]
    return len(header) >= 8 and header[4:8] == b"ftyp"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    manager = AutomationChromeManager(extension_dir=root / "extension", download_dir=output_dir)
    chrome = manager.launch_or_reuse()
    cdp = await CDPController.connect(chrome.ws_endpoint)
    started = time.time()
    try:
        await cdp.configure_downloads(str(output_dir))
        page = await cdp.get_or_create_page(GROK_IMAGINE_URL, reuse_tab=True)
        await page.bring_to_front()
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        except Exception:
            pass
        await page.wait_for_timeout(2500)

        removed = await _remove_attached_images(page)
        upload = await _upload_image(page, cdp, image_path)
        print("UPLOAD=" + json.dumps(upload, ensure_ascii=True), flush=True)
        if not upload.get("ok"):
            raise RuntimeError("Image upload did not attach in Grok composer: " + json.dumps(upload, ensure_ascii=True)[:1200])

        submit = await _submit_video_prompt(page, args.prompt, args.quality, args.duration)
        print("SUBMIT=" + json.dumps(submit, ensure_ascii=True), flush=True)
        if not submit.get("ok"):
            raise RuntimeError("Prompt submit failed: " + json.dumps(submit, ensure_ascii=True)[:1200])

        ready_state = await _wait_for_video_ready(page, args.timeout_s)
        print("READY=" + json.dumps(ready_state, ensure_ascii=True), flush=True)
        download = await _download_video(page, ready_state, output_path)

        href = str(ready_state.get("href") or page.url or "")
        match = POST_RE.search(href)
        post_url = f"https://grok.com/imagine/post/{match.group(1)}" if match else ""
        report = {
            "ok": _valid_mp4(output_path),
            "image": str(image_path),
            "output": str(output_path),
            "bytes": output_path.stat().st_size if output_path.exists() else 0,
            "postUrl": post_url,
            "download": download,
            "upload": upload,
            "removedAttachedImages": removed,
            "elapsedSeconds": round(time.time() - started, 1),
        }
        (output_dir / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        await cdp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Grok Imagine local image-to-video generation over Automation Chrome CDP.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--image", default=str(REPO_ROOT / "images" / "snap1.jpg"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-name", default="grok_cdp_image_to_video_smoke.mp4")
    parser.add_argument(
        "--prompt",
        default="Animate the uploaded image into a short cinematic video: slow camera push-in, subtle parallax, natural lighting, crisp detail, no text or captions.",
    )
    parser.add_argument("--quality", default="480p")
    parser.add_argument("--duration", default="6s")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    report = asyncio.run(run(parse_args()))
    print("REPORT=" + json.dumps(report, ensure_ascii=True), flush=True)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
