from __future__ import annotations

import json

from playwright.async_api import Browser, Page, async_playwright


class CDPController:
    def __init__(self, browser: Browser, playwright_instance):
        self.browser = browser
        self._playwright = playwright_instance

    @classmethod
    async def connect(cls, ws_endpoint: str) -> "CDPController":
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.connect_over_cdp(ws_endpoint)
        except Exception:
            await pw.stop()
            raise
        return cls(browser=browser, playwright_instance=pw)

    async def close(self) -> None:
        await self.browser.close()
        await self._playwright.stop()

    async def find_page_by_url_contains(self, substr: str) -> Page | None:
        for context in self.browser.contexts:
            for page in context.pages:
                if substr in (page.url or ""):
                    return page
        return None

    async def get_most_recent_page(self) -> Page | None:
        for context in self.browser.contexts:
            if context.pages:
                return context.pages[-1]
        return None

    async def get_or_create_page(self, url: str, reuse_tab: bool = False) -> Page:
        if reuse_tab:
            page = await self.get_most_recent_page()
            if page is not None:
                await page.goto(url)
                return page

        page = await self.find_page_by_url_contains(url)
        if page is not None:
            return page

        context = self.browser.contexts[0] if self.browser.contexts else await self.browser.new_context()
        page = await context.new_page()
        await page.goto(url)
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
