from __future__ import annotations

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

    async def get_or_create_page(self, url: str) -> Page:
        page = await self.find_page_by_url_contains(url)
        if page is not None:
            return page
        context = self.browser.contexts[0] if self.browser.contexts else await self.browser.new_context()
        page = await context.new_page()
        await page.goto(url)
        return page

    async def navigate(self, page: Page, url: str) -> None:
        await page.goto(url)

    async def smoke_test(self) -> str:
        page = await self.get_or_create_page("https://example.com")
        await page.wait_for_load_state("domcontentloaded")
        return await page.title()
