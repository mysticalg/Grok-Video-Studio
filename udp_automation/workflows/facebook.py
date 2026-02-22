from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str, title: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "facebook", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "facebook"})
    executor.run("upload.select_file", {"platform": "facebook", "filePath": video_path})
    executor.run("form.fill", {"platform": "facebook", "fields": {"title": title, "description": caption}})
    executor.run("post.submit", {"platform": "facebook"})
    return executor.run("post.status", {"platform": "facebook"})
