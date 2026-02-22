from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, title: str, description: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "youtube", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "youtube"})
    executor.run("upload.select_file", {"platform": "youtube", "filePath": video_path})
    executor.run("form.fill", {"platform": "youtube", "fields": {"title": title, "description": description}})
    executor.run("post.submit", {"platform": "youtube"})
    return executor.run("post.status", {"platform": "youtube"})
