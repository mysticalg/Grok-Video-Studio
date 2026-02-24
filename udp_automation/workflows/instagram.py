from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "instagram", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "instagram"})

    executor.run("upload.select_file", {"platform": "instagram", "filePath": video_path})
    executor.run("form.fill", {"platform": "instagram", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "instagram"})
    return executor.run("post.status", {"platform": "instagram"})

