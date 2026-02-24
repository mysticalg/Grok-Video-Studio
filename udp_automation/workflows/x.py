from __future__ import annotations

from typing import Any

from udp_automation.executors import BaseExecutor


def run(executor: BaseExecutor, video_path: str, caption: str) -> dict[str, Any]:
    executor.run("platform.open", {"platform": "x", "reuseTab": True})
    executor.run("platform.ensure_logged_in", {"platform": "x"})

    executor.run("upload.select_file", {"platform": "x", "filePath": video_path})
    executor.run("form.fill", {"platform": "x", "fields": {"description": caption}})
    executor.run("post.submit", {"platform": "x"})
    return executor.run("post.status", {"platform": "x"})
