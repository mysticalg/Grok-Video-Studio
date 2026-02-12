import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
API_BASE_URL = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")


@dataclass
class GrokConfig:
    api_key: str
    chat_model: str
    image_model: str


class GenerateWorker(QThread):
    finished_video = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, config: GrokConfig, concept: str, count: int):
        super().__init__()
        self.config = config
        self.concept = concept
        self.count = count

    def run(self) -> None:
        try:
            for idx in range(1, self.count + 1):
                self.status.emit(f"Generating variant {idx}/{self.count}...")
                video = self.generate_one_video(self.concept, idx)
                self.finished_video.emit(video)
            self.status.emit("Generation complete.")
        except Exception as exc:
            self.failed.emit(str(exc))

    def _api_error_message(self, response: requests.Response) -> str:
        try:
            return response.json().get("error", {}).get("message", response.text[:500])
        except Exception:
            return response.text[:500] or response.reason

    def call_grok_chat(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": self.config.chat_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.9,
            },
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(f"Chat request failed: {response.status_code} {self._api_error_message(response)}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def start_video_job(self, prompt: str, resolution: str) -> str:
        response = requests.post(
            f"{API_BASE_URL}/imagine/video/generations",
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.image_model,
                "prompt": prompt,
                "duration_seconds": 10,
                "resolution": resolution,
                "fps": 24,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("id")

    def poll_video_job(self, job_id: str, timeout_s: int = 420) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            response = requests.get(
                f"{API_BASE_URL}/imagine/video/generations/{job_id}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")
            if status == "succeeded":
                return payload
            if status == "failed":
                raise RuntimeError(payload.get("error", "Video generation failed"))
            time.sleep(5)
        raise TimeoutError("Timed out waiting for video generation")

    def download_video(self, video_url: str, suffix: str) -> Path:
        file_path = DOWNLOAD_DIR / f"video_{int(time.time() * 1000)}_{suffix}.mp4"
        with requests.get(video_url, stream=True, timeout=240) as response:
            response.raise_for_status()
            with open(file_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return file_path

    def generate_one_video(self, concept: str, variant: int) -> dict:
        prompt = self.call_grok_chat(
            "You write highly visual prompts for short cinematic AI videos.",
            f"Create one polished video prompt for a 10 second scene in 720p from this concept: {concept}. This is variant #{variant}.",
        )

        video_job_id = None
        chosen_resolution = None
        for resolution in ["1280x720", "640x420"]:
            try:
                video_job_id = self.start_video_job(prompt, resolution)
                chosen_resolution = resolution
                break
            except requests.HTTPError:
                continue

        if not video_job_id:
            raise RuntimeError("Could not start a video generation job")

        result = self.poll_video_job(video_job_id)
        video_url = result.get("output", {}).get("video_url") or result.get("video_url")
        if not video_url:
            raise RuntimeError("No video URL returned")

        file_path = self.download_video(video_url, f"v{variant}")
        return {
            "title": f"Generated Video {variant}",
            "prompt": prompt,
            "resolution": chosen_resolution,
            "video_file_path": str(file_path),
            "source_url": video_url,
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grok Video Desktop Studio")
        self.resize(1500, 900)
        self.videos: list[dict] = []
        self.worker: GenerateWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter()

        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Grok API Key"))
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(os.getenv("GROK_API_KEY", ""))
        left_layout.addWidget(self.api_key)

        left_layout.addWidget(QLabel("Chat Model"))
        self.chat_model = QLineEdit(os.getenv("GROK_CHAT_MODEL", "grok-3-mini"))
        left_layout.addWidget(self.chat_model)

        left_layout.addWidget(QLabel("Video Model"))
        self.image_model = QLineEdit(os.getenv("GROK_VIDEO_MODEL", "grok-video-latest"))
        left_layout.addWidget(self.image_model)

        left_layout.addWidget(QLabel("Concept"))
        self.concept = QPlainTextEdit()
        self.concept.setPlaceholderText("Describe the video idea...")
        left_layout.addWidget(self.concept)

        row = QHBoxLayout()
        row.addWidget(QLabel("Count"))
        self.count = QSpinBox()
        self.count.setRange(1, 10)
        self.count.setValue(1)
        row.addWidget(self.count)
        left_layout.addLayout(row)

        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.clicked.connect(self.start_generation)
        left_layout.addWidget(self.generate_btn)

        self.open_btn = QPushButton("Open Local Video...")
        self.open_btn.clicked.connect(self.open_local_video)
        left_layout.addWidget(self.open_btn)

        left_layout.addWidget(QLabel("Generated Videos"))
        self.video_picker = QComboBox()
        self.video_picker.currentIndexChanged.connect(self.show_selected_video)
        left_layout.addWidget(self.video_picker)

        left_layout.addWidget(QLabel("Activity Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        left_layout.addWidget(self.log)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://grok.com"))

        splitter.addWidget(left)
        splitter.addWidget(self.browser)
        splitter.setSizes([500, 1000])

        # Keep browser visible as a fixed right-hand pane
        splitter.setChildrenCollapsible(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def start_generation(self) -> None:
        api_key = self.api_key.text().strip()
        concept = self.concept.toPlainText().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Please enter a Grok API key.")
            return
        if not concept:
            QMessageBox.warning(self, "Missing Concept", "Please enter a concept.")
            return

        config = GrokConfig(
            api_key=api_key,
            chat_model=self.chat_model.text().strip() or "grok-3-mini",
            image_model=self.image_model.text().strip() or "grok-video-latest",
        )

        self.generate_btn.setEnabled(False)
        self.worker = GenerateWorker(config, concept, self.count.value())
        self.worker.status.connect(self._append_log)
        self.worker.finished_video.connect(self.on_video_finished)
        self.worker.failed.connect(self.on_generation_error)
        self.worker.finished.connect(lambda: self.generate_btn.setEnabled(True))
        self.worker.start()

    def on_video_finished(self, video: dict) -> None:
        self.videos.append(video)
        label = f"{video['title']} ({video['resolution']})"
        self.video_picker.addItem(label)
        self.video_picker.setCurrentIndex(self.video_picker.count() - 1)
        self._append_log(f"Saved: {video['video_file_path']}")

    def on_generation_error(self, error: str) -> None:
        self._append_log(f"ERROR: {error}")
        QMessageBox.critical(self, "Generation Failed", error)

    def show_selected_video(self, index: int) -> None:
        if index < 0 or index >= len(self.videos):
            return
        video = self.videos[index]
        self.browser.setUrl(QUrl.fromLocalFile(video["video_file_path"]))

    def open_local_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video", str(DOWNLOAD_DIR), "Videos (*.mp4 *.mov *.webm)")
        if not file_path:
            return
        self.browser.setUrl(QUrl.fromLocalFile(file_path))
        self._append_log(f"Opened local file: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
