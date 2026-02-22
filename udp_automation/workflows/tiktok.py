from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "tiktok", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "tiktok"})
    executor.run("upload.select_file", {"platform": "tiktok", "filePath": video_path})
    executor.run("form.fill", {"platform": "tiktok", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "tiktok"})
    return executor.run("post.status", {"platform": "tiktok"})
