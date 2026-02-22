from __future__ import annotations

import json

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


class CDPController:
    def __init__(self, browser: Browser | None, playwright_instance, context: BrowserContext | None = None):
        self.browser = browser
        self.context = context
        self._playwright = playwright_instance

    @classmethod
    async def connect(cls, ws_endpoint: str) -> "CDPController":
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.connect_over_cdp(ws_endpoint)
        except Exception:
            await pw.stop()
            raise
        return cls(browser=browser, context=None, playwright_instance=pw)

    @classmethod
    async def launch_persistent(
        cls,
        *,
        user_data_dir: str,
        extension_dir: str,
        executable_path: str | None = None,
        headless: bool = False,
    ) -> "CDPController":
        pw = await async_playwright().start()
        args = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
            "--disable-features=Translate",
            f"--disable-extensions-except={extension_dir}",
            f"--load-extension={extension_dir}",
        ]
        try:
            context = await pw.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=headless,
                executable_path=executable_path,
                args=args,
                ignore_default_args=["--disable-extensions"],
            )
        except Exception:
            await pw.stop()
            raise
        return cls(browser=context.browser, context=context, playwright_instance=pw)

    async def close(self) -> None:
        if self.context is not None:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
        elif self.browser is not None:
            try:
                await self.browser.close()
            except Exception:
                pass
        self.browser = None
        await self._playwright.stop()

    def _iter_contexts(self) -> list[BrowserContext]:
        if self.context is not None:
            return [self.context]
        if self.browser is None:
            return []
        return list(self.browser.contexts)

    async def find_page_by_url_contains(self, substr: str) -> Page | None:
        for context in self._iter_contexts():
            for page in context.pages:
                if substr in (page.url or ""):
                    return page
        return None

    async def get_most_recent_page(self) -> Page | None:
        contexts = self._iter_contexts()
        for context in contexts:
            if context.pages:
                return context.pages[-1]
        return None

    async def _goto_best_effort(self, page: Page, url: str) -> None:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception:
            pass

    async def get_or_create_page(self, url: str, reuse_tab: bool = False) -> Page:
        if reuse_tab:
            page = await self.get_most_recent_page()
            if page is not None:
                if url not in (page.url or ""):
                    await self._goto_best_effort(page, url)
                return page

        page = await self.find_page_by_url_contains(url)
        if page is not None:
            return page

        if self.context is not None:
            context = self.context
        elif self.browser is not None and self.browser.contexts:
            context = self.browser.contexts[0]
        elif self.browser is not None:
            context = await self.browser.new_context()
        else:
            raise RuntimeError("Browser context is not available")

        page = await context.new_page()
        await self._goto_best_effort(page, url)
        return page

    async def navigate(self, page: Page, url: str) -> None:
        await page.goto(url)

    async def set_file_input_files_via_dom(self, page: Page, selector: str, file_path: str) -> bool:
        session = await page.context.new_cdp_session(page)
        expression = f"document.querySelector({json.dumps(selector)})"
        evaluate = await session.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": False},
        )
        remote_obj = evaluate.get("result") or {}
        object_id = remote_obj.get("objectId")
        if not object_id:
            return False

        try:
            node_info = await session.send("DOM.requestNode", {"objectId": object_id})
            node_id = node_info.get("nodeId")
            if not node_id:
                return False
            await session.send("DOM.setFileInputFiles", {"nodeId": node_id, "files": [file_path]})
            return True
        finally:
            try:
                await session.send("Runtime.releaseObject", {"objectId": object_id})
            except Exception:
                pass

    async def smoke_test(self) -> str:
        page = await self.get_or_create_page("https://example.com")
        await page.wait_for_load_state("domcontentloaded")
        return await page.title()
