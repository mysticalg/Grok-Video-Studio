import os
import asyncio
import json
import re
import base64
import concurrent.futures
import hashlib
import secrets
import shutil
import subprocess
import string
import sys
import tempfile
import threading
import time
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse
from typing import Any, Callable, Iterable

import requests
from PySide6.QtCore import QEvent, QMimeData, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCloseEvent, QDesktopServices, QGuiApplication, QIcon, QImage, QPainter, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QLayout,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QStatusBar,
    QStyle,
    QScrollArea,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QHeaderView,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from social_uploaders import upload_video_to_facebook_page, upload_video_to_instagram_reels, upload_video_to_tiktok
from youtube_uploader import upload_video_to_youtube
from grok_web_automation import build_trained_process, run_trained_process, train_browser_flow
from automation.chrome_manager import AutomationChromeManager, ChromeInstance
from automation.cdp_controller import CDPController
from automation.control_bus import ControlBusServer
from udp_automation.executors import UdpExecutor
from udp_automation.service import UdpAutomationService
from udp_automation.workflows import facebook as udp_facebook_workflow
from udp_automation.workflows import tiktok as udp_tiktok_workflow
from udp_automation.workflows import youtube as udp_youtube_workflow

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR = DOWNLOAD_DIR / ".thumbnails"
THUMBNAILS_DIR.mkdir(exist_ok=True)
CACHE_DIR = BASE_DIR / ".qtwebengine"
QTWEBENGINE_USE_DISK_CACHE = True
MIN_VALID_VIDEO_BYTES = 1 * 1024 * 1024
API_BASE_URL = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_OAUTH_ISSUER = os.getenv("OPENAI_OAUTH_ISSUER", "https://auth.openai.com")
OPENAI_CODEX_CLIENT_ID = os.getenv("OPENAI_CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
OPENAI_OAUTH_SCOPE = "openid profile email offline_access"
OPENAI_OAUTH_CALLBACK_PORT = int(os.getenv("OPENAI_OAUTH_CALLBACK_PORT", "1455"))
OPENAI_TOKEN_PATHS = ("/token", "/oauth/token")
OPENAI_CHATGPT_API_BASE = os.getenv("OPENAI_CHATGPT_API_BASE", "https://chatgpt.com/backend-api/codex")
OPENAI_USE_CHATGPT_BACKEND = os.getenv("OPENAI_USE_CHATGPT_BACKEND", "1").strip().lower() not in {"0", "false", "no"}
SEEDANCE_API_BASE = os.getenv("SEEDANCE_API_BASE", "https://api.seedance.ai/v2")
FACEBOOK_GRAPH_VERSION = os.getenv("FACEBOOK_GRAPH_VERSION", "v21.0")
FACEBOOK_OAUTH_AUTHORIZE_URL = f"https://www.facebook.com/{FACEBOOK_GRAPH_VERSION}/dialog/oauth"
FACEBOOK_OAUTH_TOKEN_URL = f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/oauth/access_token"
FACEBOOK_OAUTH_CALLBACK_PORT = int(os.getenv("FACEBOOK_OAUTH_CALLBACK_PORT", "1456"))
FACEBOOK_OAUTH_SCOPE = os.getenv(
    "FACEBOOK_OAUTH_SCOPE",
    "pages_show_list,pages_manage_posts,publish_video",
)
TIKTOK_OAUTH_AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TIKTOK_OAUTH_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TIKTOK_OAUTH_CALLBACK_PORT = int(os.getenv("TIKTOK_OAUTH_CALLBACK_PORT", "1457"))
TIKTOK_OAUTH_SCOPE = os.getenv("TIKTOK_OAUTH_SCOPE", "user.info.basic,video.upload,video.publish")
TIKTOK_PKCE_CHALLENGE_ENCODING = os.getenv("TIKTOK_PKCE_CHALLENGE_ENCODING", "hex").strip().lower()
if TIKTOK_PKCE_CHALLENGE_ENCODING not in {"hex", "base64url"}:
    TIKTOK_PKCE_CHALLENGE_ENCODING = "hex"
DEFAULT_PREFERENCES_FILE = BASE_DIR / "preferences.json"
CDP_RELAY_SOCIAL_UPLOAD_URL = os.getenv("GROK_CDP_RELAY_SOCIAL_UPLOAD_URL", "http://127.0.0.1:8765/social-upload-step")
CDP_RELAY_TIMEOUT_SECONDS = max(1.0, float(os.getenv("GROK_CDP_RELAY_TIMEOUT_SECONDS", "25")))
GITHUB_REPO_URL = "https://github.com/mysticalg/Grok-video-to-youtube-api"
GITHUB_RELEASES_URL = "https://github.com/mysticalg/Grok-video-to-youtube-api/releases"
GITHUB_LATEST_RELEASE_API_URL = "https://api.github.com/repos/mysticalg/Grok-video-to-youtube-api/releases/latest"
GITHUB_ACTIONS_RUNS_URL = f"{GITHUB_REPO_URL}/actions"
BUY_ME_A_COFFEE_URL = "https://buymeacoffee.com/dhooksterm"
PAYPAL_DONATION_URL = "https://www.paypal.com/paypalme/dhookster"
SOL_DONATION_ADDRESS = "6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp"
DEFAULT_MANUAL_PROMPT_TEXT = (
    "abstract surreal artistic photorealistic strange random dream like scifi fast moving camera, "
    "fast moving fractals morphing and intersecting, highly detailed"
)

_session_download_counter_lock = threading.Lock()
_session_download_counter = 0


def _read_local_app_version(default: str = "0.0.0") -> str:
    try:
        version_path = BASE_DIR / "VERSION"
        version_text = version_path.read_text(encoding="utf-8").strip()
        return version_text or default
    except Exception:
        return default


CURRENT_APP_VERSION = _read_local_app_version()


def _next_session_download_count() -> int:
    global _session_download_counter
    with _session_download_counter_lock:
        _session_download_counter += 1
        return _session_download_counter


def _slugify_filename_part(value: str, fallback: str = "na") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return cleaned or fallback


def _normalize_version_key(value: str) -> tuple[int, ...]:
    cleaned = str(value or "").strip().lower().lstrip("v")
    if not cleaned:
        return (0,)
    parts: list[int] = []
    for token in re.split(r"[^0-9]+", cleaned):
        if token:
            try:
                parts.append(int(token))
            except ValueError:
                parts.append(0)
    return tuple(parts) if parts else (0,)


def _parse_query_preserving_plus(query: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in query.split("&"):
        if not part:
            continue
        key, sep, value = part.partition("=")
        decoded_key = unquote(key)
        decoded_value = unquote(value) if sep else ""
        if decoded_key and decoded_key not in result:
            result[decoded_key] = decoded_value
    return result

def _parse_json_object_from_text(raw: str) -> dict:
    """Parse a JSON object from a model response that may include wrappers."""
    text = (raw or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty AI response", raw, 0)

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("AI response did not contain a JSON object", text, 0)


def _decode_openai_access_token(access_token: str) -> dict:
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_claim_string(source: object, keys: tuple[str, ...]) -> str:
    if not isinstance(source, dict):
        return ""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_openai_org_project_ids(access_token: str) -> tuple[str, str]:
    claims = _decode_openai_access_token(access_token)
    auth_claim = claims.get("https://api.openai.com/auth")

    org_id = _extract_claim_string(auth_claim, ("organization_id", "org_id", "organization"))
    project_id = _extract_claim_string(auth_claim, ("project_id", "project"))

    if not org_id:
        org_id = _extract_claim_string(claims, ("organization_id", "org_id", "organization"))
    if not project_id:
        project_id = _extract_claim_string(claims, ("project_id", "project"))

    if not org_id:
        organizations = claims.get("https://api.openai.com/organizations")
        if isinstance(organizations, list):
            for item in organizations:
                org_id = _extract_claim_string(item, ("id", "organization_id", "org_id"))
                if org_id:
                    break

    return org_id, project_id


def _openai_headers_from_credential(credential: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {credential}",
        "Content-Type": "application/json",
    }
    # OAuth tokens may contain organization/project hints in claims. API keys do not.
    org_id, project_id = _extract_openai_org_project_ids(credential)
    if org_id:
        headers["OpenAI-Organization"] = org_id
    if project_id:
        headers["OpenAI-Project"] = project_id
    return headers


def _openai_chat_target(credential: str) -> tuple[str, dict[str, str], bool]:
    claims = _decode_openai_access_token(credential)
    issuer = str(claims.get("iss", "")).strip()
    auth_claim = claims.get("https://api.openai.com/auth")

    headers = _openai_headers_from_credential(credential)

    use_chatgpt_backend = (
        OPENAI_USE_CHATGPT_BACKEND
        and OPENAI_API_BASE == "https://api.openai.com/v1"
        and issuer == "https://auth.openai.com"
    )
    if use_chatgpt_backend:
        account_id = _extract_claim_string(auth_claim, ("chatgpt_account_id",))
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return f"{OPENAI_CHATGPT_API_BASE}/responses", headers, True

    return f"{OPENAI_API_BASE}/chat/completions", headers, False


def _openai_is_likely_api_key(credential: str) -> bool:
    value = credential.strip()
    if not value:
        return False
    return value.startswith("sk-") or value.startswith("rk-")




def _extract_text_from_responses_body(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    output_text = str(body.get("output_text", "")).strip()
    if output_text:
        return output_text

    text_chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict):
                text_value = str(content.get("text", "")).strip()
                if text_value:
                    text_chunks.append(text_value)
    return "\n".join(text_chunks).strip()


def _extract_text_from_responses_sse(raw_text: str) -> str:
    text_chunks: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            event = json.loads(data_str)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "response.output_text.delta":
            delta = str(event.get("delta", ""))
            if delta:
                text_chunks.append(delta)
            continue
        text_value = _extract_text_from_responses_body(event)
        if text_value:
            text_chunks.append(text_value)

    return "".join(text_chunks).strip()

def _openai_chat_payload(model: str, system: str, user: str, temperature: float) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }


def _openai_responses_payload(model: str, system: str, user: str) -> dict[str, object]:
    return {
        "model": model,
        "instructions": system,
        "input": [
            {"role": "user", "content": user},
        ],
        "store": False,
        "stream": True,
    }


def _openai_error_prefers_responses(response: requests.Response) -> bool:
    if response.status_code not in {400, 404, 422}:
        return False
    text = (response.text or "").lower()
    return "only supported in v1/responses" in text or "not in v1/chat/completions" in text


def _call_openai_chat_api(credential: str, model: str, system: str, user: str, temperature: float) -> str:
    endpoint, headers, is_responses_api = _openai_chat_target(credential)

    payload = (
        _openai_responses_payload(model, system, user)
        if is_responses_api
        else _openai_chat_payload(model, system, user, temperature)
    )

    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)

    if not response.ok and not is_responses_api and _openai_error_prefers_responses(response):
        responses_endpoint = f"{OPENAI_API_BASE}/responses"
        retry_payload = _openai_responses_payload(model, system, user)
        retry_response = requests.post(responses_endpoint, headers=headers, json=retry_payload, timeout=90)
        if not retry_response.ok:
            raise RuntimeError(
                f"OpenAI request failed: {retry_response.status_code} {retry_response.text[:500]}"
            )
        response = retry_response
        is_responses_api = True

    if not response.ok:
        raise RuntimeError(f"OpenAI request failed: {response.status_code} {response.text[:500]}")

    if is_responses_api:
        streamed_text = _extract_text_from_responses_sse(response.text)
        if streamed_text:
            return streamed_text
        body = response.json() if response.content else {}
        text_from_body = _extract_text_from_responses_body(body)
        if text_from_body:
            return text_from_body
        raise RuntimeError("OpenAI response did not include text output.")

    body = response.json()
    return str(body["choices"][0]["message"]["content"]).strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_qtwebengine_remote_debug_port_from_preferences(default: int = 0) -> int:
    if not DEFAULT_PREFERENCES_FILE.exists():
        return default

    try:
        with open(DEFAULT_PREFERENCES_FILE, "r", encoding="utf-8") as handle:
            preferences = json.load(handle)
    except Exception:
        return default

    if not isinstance(preferences, dict):
        return default

    enabled = preferences.get("qtwebengine_remote_debug_enabled", False)
    if not enabled:
        return default

    try:
        port = int(preferences.get("qtwebengine_remote_debug_port", default) or 0)
    except (TypeError, ValueError):
        return default

    return port if port > 0 else default


def _path_supports_rw(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / f".rw_probe_{os.getpid()}"
        moved_file = path / f".rw_probe_{os.getpid()}_moved"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.replace(moved_file)
        moved_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_qtwebengine_cache_dir() -> tuple[Path, bool]:
    candidates: list[Path] = []

    env_cache_dir = os.getenv("GROK_BROWSER_CACHE_DIR", "").strip()
    if env_cache_dir:
        candidates.append(Path(env_cache_dir).expanduser())

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        candidates.append(Path(local_app_data) / "GrokVideoDesktopStudio" / "qtwebengine")

    xdg_cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home) / "GrokVideoDesktopStudio" / "qtwebengine")

    candidates.append(Path.home() / "Library" / "Caches" / "GrokVideoDesktopStudio" / "qtwebengine")
    candidates.append(Path.home() / ".cache" / "GrokVideoDesktopStudio" / "qtwebengine")

    candidates.append(BASE_DIR / ".qtwebengine")
    candidates.append(Path(tempfile.gettempdir()) / "GrokVideoDesktopStudio" / "qtwebengine")

    for candidate in candidates:
        if _path_supports_rw(candidate):
            return candidate, True

    fallback = Path(tempfile.gettempdir()) / f"GrokVideoDesktopStudio_qtwebengine_{os.getpid()}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, False




def _probe_video_color_properties(video_path: str | Path) -> dict[str, str]:
    path_str = str(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_range,color_space,color_transfer,color_primaries,pix_fmt,width,height",
        "-of",
        "json",
        path_str,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return {}

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return {}

    stream = streams[0] if isinstance(streams[0], dict) else {}
    normalized: dict[str, str] = {}
    for key in ("color_range", "color_space", "color_transfer", "color_primaries", "pix_fmt"):
        value = str(stream.get(key) or "").strip().lower()
        if value and value not in {"unknown", "unspecified", "n/a"}:
            normalized[key] = value

    width = stream.get("width")
    height = stream.get("height")
    if isinstance(width, int):
        normalized["width"] = str(width)
    if isinstance(height, int):
        normalized["height"] = str(height)
    return normalized


def _probe_video_duration_seconds(video_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return max(0.0, float((result.stdout or "0").strip() or "0"))


def _seconds_to_srt_time(value: float) -> str:
    clamped_ms = max(0, int(round(value * 1000)))
    hours, rem = divmod(clamped_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _escape_srt_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    return cleaned.replace("-->", "→")



def _build_last_frame_extraction_cmds(video_path: str | Path, seek_args: list[str], frame_path: Path) -> list[list[str]]:
    color_meta = _probe_video_color_properties(video_path)
    matrix = "bt709"
    range_raw = color_meta.get("color_range", "")
    in_range = "full" if range_raw in {"pc", "jpeg", "full"} else "limited"

    color_filter = (
        f"scale=in_range={in_range}:out_range=full:"
        f"in_color_matrix={matrix}:out_color_matrix={matrix},format=rgb48le"
    )

    path_str = str(video_path)
    frame_str = str(frame_path)
    return [
        [
            "ffmpeg",
            "-y",
            *seek_args,
            "-i",
            path_str,
            "-frames:v",
            "1",
            frame_str,
        ],
        [
            "ffmpeg",
            "-y",
            *seek_args,
            "-i",
            path_str,
            "-vf",
            color_filter,
            "-frames:v",
            "1",
            frame_str,
        ],
        [
            "ffmpeg",
            "-y",
            *seek_args,
            "-i",
            path_str,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb48be",
            "-sws_flags",
            "accurate_rnd+full_chroma_int",
            frame_str,
        ],
        [
            "ffmpeg",
            "-y",
            *seek_args,
            "-i",
            path_str,
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgba",
            frame_str,
        ],
        [
            "ffmpeg",
            "-y",
            *seek_args,
            "-i",
            path_str,
            "-frames:v",
            "1",
            frame_str,
        ],
    ]

def _configure_qtwebengine_runtime() -> None:
    global CACHE_DIR, QTWEBENGINE_USE_DISK_CACHE
    CACHE_DIR, QTWEBENGINE_USE_DISK_CACHE = _resolve_qtwebengine_cache_dir()

    default_flags = [
        "--enable-gpu-rasterization",
        "--enable-zero-copy",
        "--ignore-gpu-blocklist",
        "--disable-renderer-backgrounding",
        "--autoplay-policy=no-user-gesture-required",
    ]

    existing_flags = os.getenv("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(default_flags + ([existing_flags] if existing_flags else []))



@dataclass
class GrokConfig:
    api_key: str
    chat_model: str
    image_model: str


@dataclass
class PromptConfig:
    prompt_source: str
    video_provider: str
    concept: str
    manual_prompt: str
    openai_api_key: str
    openai_access_token: str
    openai_chat_model: str
    video_resolution: str
    video_resolution_label: str
    video_aspect_ratio: str
    video_duration_seconds: int
    openai_sora_settings: dict[str, object]
    seedance_settings: dict[str, object]


@dataclass
class AISocialMetadata:
    title: str
    medium_title: str
    tiktok_subheading: str
    description: str
    hashtags: list[str]
    category: str


class GenerateWorker(QThread):
    finished_video = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, config: GrokConfig, prompt_config: PromptConfig, count: int, download_dir: Path):
        super().__init__()
        self.config = config
        self.prompt_config = prompt_config
        self.count = count
        self.download_dir = download_dir
        self.stop_requested = False
        self._openai_uploaded_reference_cache: dict[str, str] = {}
        self._openai_last_generated_video_path: Path | None = None
        self._openai_last_generated_reference_id: str = ""

    def _compose_output_filename(self, prefix: str, variant: int, extension: str = "mp4") -> str:
        provider = _slugify_filename_part(self.prompt_config.video_provider or "video")
        resolution = _slugify_filename_part(self.prompt_config.video_resolution or "auto")
        aspect = _slugify_filename_part(self.prompt_config.video_aspect_ratio or "na")
        session_index = _next_session_download_count()
        return f"{prefix}_{provider}_{resolution}_{aspect}_v{variant:02d}_d{session_index:03d}.{extension}"

    def request_stop(self) -> None:
        self.stop_requested = True
        self.requestInterruption()

    def _ensure_not_stopped(self) -> None:
        if self.stop_requested or self.isInterruptionRequested():
            raise RuntimeError("Generation stopped by user")

    def run(self) -> None:
        try:
            for idx in range(1, self.count + 1):
                self._ensure_not_stopped()
                self.status.emit(f"Generating variant {idx}/{self.count}...")
                video = self.generate_one_video(idx)
                self.finished_video.emit(video)
            self.status.emit("Generation complete.")
        except Exception as exc:
            if str(exc) == "Generation stopped by user":
                self.status.emit("Generation stopped.")
                return
            self.failed.emit(str(exc))

    def _api_error_message(self, response: requests.Response) -> str:
        try:
            return response.json().get("error", {}).get("message", response.text[:500])
        except Exception:
            return response.text[:500] or response.reason

    def _openai_credential(self) -> str:
        return self.prompt_config.openai_api_key or self.prompt_config.openai_access_token

    def _openai_video_headers(self) -> dict[str, str]:
        credential = self._openai_credential()
        if not credential:
            raise RuntimeError("OpenAI API key or access token is required.")
        return _openai_headers_from_credential(credential)

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

    def call_openai_chat(self, system: str, user: str) -> str:
        openai_credential = self._openai_credential()
        if not openai_credential:
            raise RuntimeError("OpenAI API key or access token is required.")
        return _call_openai_chat_api(
            credential=openai_credential,
            model=self.prompt_config.openai_chat_model,
            system=system,
            user=user,
            temperature=0.9,
        )

    def build_prompt(self, variant: int) -> str:
        self._ensure_not_stopped()
        source = self.prompt_config.prompt_source
        if source == "manual":
            return self.prompt_config.manual_prompt

        system = "You write highly visual prompts for short cinematic AI videos."
        user = (
            "Create one polished video prompt for a "
            f"{self.prompt_config.video_duration_seconds} second scene in "
            f"{self.prompt_config.video_resolution_label} with a {self.prompt_config.video_aspect_ratio} aspect ratio "
            f"from this concept: {self.prompt_config.concept}. This is variant #{variant}."
        )

        if source == "openai":
            return self.call_openai_chat(system, user)
        return self.call_grok_chat(system, user)

    def start_video_job(self, prompt: str, resolution: str, duration_seconds: int) -> str:
        self._ensure_not_stopped()
        response = requests.post(
            f"{API_BASE_URL}/imagine/video/generations",
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.image_model,
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "resolution": resolution,
                "fps": 24,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("id")

    def _openai_sora_payload_variants(self, prompt: str) -> list[dict[str, object]]:
        settings = self.prompt_config.openai_sora_settings if isinstance(self.prompt_config.openai_sora_settings, dict) else {}
        model = str(settings.get("model") or "sora-2")
        size = str(settings.get("size") or self.prompt_config.video_resolution or "1280x720")
        seconds = str(settings.get("seconds") or str(self.prompt_config.video_duration_seconds or 8))
        input_reference = self._resolve_openai_input_reference(settings)
        extra_body = settings.get("extra_body") if isinstance(settings.get("extra_body"), dict) else {}

        base: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }
        if input_reference:
            base["input_reference"] = input_reference
        if extra_body:
            base.update(extra_body)

        with_seconds = dict(base)
        with_seconds["seconds"] = seconds
        return [with_seconds, base]

    def _openai_setting_enabled(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _looks_like_openai_file_id(self, value: str) -> bool:
        normalized = str(value).strip()
        return normalized.startswith("file-") and len(normalized) > len("file-")

    def _extract_openai_input_reference_id(self, payload: object) -> str:
        candidates: list[str] = []

        def collect(value: object) -> None:
            if isinstance(value, dict):
                for key in ("image_id", "input_reference", "input_image_id", "reference_image_id", "file_id"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        candidates.append(candidate.strip())
                output = value.get("output")
                if isinstance(output, dict):
                    collect(output)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(payload)
        return candidates[0] if candidates else ""

    def _openai_input_reference_payload_variants(self, payload: dict[str, object]) -> list[dict[str, object]]:
        # OpenAI Sora expects `input_reference` as a file value (string file id in this integration).
        # Object forms such as {"file_id": ...} or {"id": ...} trigger invalid_type errors.
        return [payload]

    def _resolve_latest_video_for_sora_continuation(self) -> Path | None:
        video_extensions = {".mp4", ".mov", ".m4v", ".webm"}
        if self._openai_last_generated_video_path and self._openai_last_generated_video_path.exists():
            return self._openai_last_generated_video_path

        files = [
            path
            for path in self.download_dir.glob("*")
            if path.is_file() and path.suffix.lower() in video_extensions
        ]
        if not files:
            return None
        return max(files, key=lambda path: path.stat().st_mtime)

    def _extract_last_frame_for_sora(self, video_path: Path) -> Path:
        self._ensure_not_stopped()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        extraction_attempts = [
            ["-sseof", "-0.05"],
            ["-sseof", "-0.5"],
            ["-sseof", "-1.0"],
        ]
        failure_messages: list[str] = []

        for attempt_index, seek_args in enumerate(extraction_attempts):
            frame_path = self.download_dir / f"sora_last_frame_{int(time.time() * 1000)}_{attempt_index}.png"
            extraction_cmds = _build_last_frame_extraction_cmds(
                str(video_path),
                list(seek_args),
                frame_path,
            )

            result = None
            for cmd in extraction_cmds:
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    break
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        "ffmpeg is required for OpenAI Sora last-frame continuity but was not found in PATH."
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    failure_messages.append(exc.stderr[-500:] or "ffmpeg failed")
                    result = None

            if frame_path.exists() and frame_path.stat().st_size > 0:
                return frame_path

            if result is not None:
                failure_messages.append(result.stderr[-500:] or "ffmpeg produced an empty frame output")

        raise RuntimeError(
            "Sora continuity frame extraction produced an empty image. "
            f"Attempts failed with: {' | '.join(msg.strip() for msg in failure_messages if msg.strip())[:1200]}"
        )

    def _upload_openai_input_reference_file(self, file_path: Path) -> str:
        self._ensure_not_stopped()
        cache_key = str(file_path.resolve())
        if cache_key in self._openai_uploaded_reference_cache:
            return self._openai_uploaded_reference_cache[cache_key]

        headers = self._openai_video_headers()
        upload_headers = {key: value for key, value in headers.items() if key.lower() != "content-type"}

        with file_path.open("rb") as handle:
            response = requests.post(
                f"{OPENAI_API_BASE}/files",
                headers=upload_headers,
                data={"purpose": "vision"},
                files={"file": (file_path.name, handle, "application/octet-stream")},
                timeout=120,
            )

        if not response.ok:
            raise RuntimeError(f"OpenAI file upload failed for input_reference: {response.status_code} {response.text[:500]}")

        payload = response.json() if response.content else {}
        file_id = str(payload.get("id") or "").strip()
        if not file_id:
            raise RuntimeError("OpenAI file upload succeeded but did not return a file id.")

        self._openai_uploaded_reference_cache[cache_key] = file_id
        return file_id

    def _resolve_openai_input_reference(self, settings: dict[str, object]) -> str:
        input_reference = str(settings.get("input_reference") or "").strip()

        if self._openai_setting_enabled(settings.get("continue_from_last_frame")):
            if self._openai_last_generated_reference_id and self._looks_like_openai_file_id(self._openai_last_generated_reference_id):
                self.status.emit(
                    f"Sora continuity: reusing previous OpenAI file reference id {self._openai_last_generated_reference_id}."
                )
                input_reference = self._openai_last_generated_reference_id
            else:
                if self._openai_last_generated_reference_id:
                    self.status.emit(
                        "Sora continuity: previous OpenAI reference id is not a file id; extracting and uploading a frame instead."
                    )
                latest_video = self._resolve_latest_video_for_sora_continuation()
                if latest_video is None:
                    raise RuntimeError(
                        "Sora continuity is enabled, but no generated videos were found in the downloads folder to extract from."
                    )
                self.status.emit(f"Sora continuity: extracting last frame from {latest_video.name}...")
                frame_path = self._extract_last_frame_for_sora(latest_video)
                input_reference = str(frame_path)

        if not input_reference:
            return ""

        local_path = Path(input_reference)
        if local_path.exists() and local_path.is_file():
            self.status.emit(f"Sora continuity: using local input reference file {local_path.name}.")
            return str(local_path)

        return input_reference

    def _post_openai_sora_generation_request(self, endpoint: str, headers: dict[str, str], payload: dict[str, object]) -> requests.Response:
        input_reference_value = payload.get("input_reference")
        if isinstance(input_reference_value, str):
            candidate_path = Path(input_reference_value)
            if candidate_path.exists() and candidate_path.is_file():
                request_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
                data = {k: str(v) for k, v in payload.items() if k != "input_reference"}
                with candidate_path.open("rb") as handle:
                    return requests.post(
                        endpoint,
                        headers=request_headers,
                        data=data,
                        files={"input_reference": (candidate_path.name, handle, "image/png")},
                        timeout=90,
                    )
        return requests.post(endpoint, headers=headers, json=payload, timeout=90)

    def _start_openai_video_job(self, prompt: str) -> tuple[str | None, str | None, str]:
        self._ensure_not_stopped()
        headers = self._openai_video_headers()

        endpoints = [
            f"{OPENAI_API_BASE}/videos/generations",
            f"{OPENAI_API_BASE}/videos",
        ]

        payload_variants = self._openai_sora_payload_variants(prompt)
        last_error = ""
        for endpoint in endpoints:
            for idx, payload in enumerate(payload_variants):
                request_payload_variants = self._openai_input_reference_payload_variants(payload)
                for payload_variant_idx, request_payload in enumerate(request_payload_variants):
                    response = self._post_openai_sora_generation_request(endpoint, headers, request_payload)
                    if response.ok:
                        data = response.json() if response.content else {}
                        job_id = data.get("id") or data.get("job_id")
                        if not job_id and isinstance(data.get("data"), list) and data["data"]:
                            job_id = data["data"][0].get("id")
                        direct_url = (
                            data.get("video_url")
                            or data.get("url")
                            or (data.get("output") or {}).get("video_url")
                        )
                        reference_id = self._extract_openai_input_reference_id(data)
                        return job_id, direct_url, reference_id

                    status = response.status_code
                    body_text = response.text[:500]
                    body_lower = body_text.lower()
                    if status in {404, 405, 501}:
                        last_error = body_text[:400]
                        break

                    input_reference_type_error = (
                        status in {400, 422}
                        and "input_reference" in body_lower
                        and (
                            "invalid type" in body_lower
                            or "unknown parameter" in body_lower
                            or "expected" in body_lower
                        )
                    )
                    if input_reference_type_error:
                        last_error = body_text[:400]

                    seconds_error = (
                        status in {400, 422}
                        and "seconds" in body_lower
                        and (
                            "unknown parameter" in body_lower
                            or "invalid type" in body_lower
                            or "invalid value" in body_lower
                            or "expected one of" in body_lower
                        )
                    )
                    if seconds_error and idx < len(payload_variants) - 1 and payload_variant_idx == len(request_payload_variants) - 1:
                        last_error = body_text[:400]
                        break

                    raise RuntimeError(f"OpenAI Sora request failed: {status} {body_text}")

        raise RuntimeError(
            "OpenAI Sora API endpoint not available for this account/base URL. "
            f"Last error: {last_error or 'none'}"
        )

    def _poll_openai_video_job(self, job_id: str, timeout_s: int = 900) -> dict:
        self._ensure_not_stopped()
        headers = self._openai_video_headers()
        endpoints = [
            f"{OPENAI_API_BASE}/videos/generations/{job_id}",
            f"{OPENAI_API_BASE}/videos/{job_id}",
        ]

        start = time.time()
        while time.time() - start < timeout_s:
            self._ensure_not_stopped()
            for endpoint in endpoints:
                response = requests.get(endpoint, headers=headers, timeout=60)
                if response.status_code in {404, 405, 501}:
                    continue
                if not response.ok:
                    raise RuntimeError(f"OpenAI Sora poll failed: {response.status_code} {response.text[:500]}")

                payload = response.json() if response.content else {}
                status = str(payload.get("status") or payload.get("state") or "").lower()
                if status in {"succeeded", "completed", "ready"}:
                    return payload
                if status in {"failed", "error", "cancelled", "canceled"}:
                    error_msg = payload.get("error") or payload.get("last_error") or "Video generation failed"
                    raise RuntimeError(f"OpenAI Sora generation failed: {error_msg}")
            time.sleep(5)

        raise TimeoutError("Timed out waiting for OpenAI Sora video generation")

    def _extract_video_url_from_payload(self, payload: dict) -> str:
        candidates = [
            payload.get("video_url"),
            payload.get("url"),
            (payload.get("output") or {}).get("video_url") if isinstance(payload.get("output"), dict) else None,
        ]
        if isinstance(payload.get("data"), list):
            for item in payload.get("data"):
                if isinstance(item, dict):
                    candidates.extend([item.get("video_url"), item.get("url")])
                    if isinstance(item.get("output"), dict):
                        candidates.append(item["output"].get("video_url"))

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    def _download_openai_video_content(self, job_id: str, suffix: str) -> Path:
        self._ensure_not_stopped()
        headers = self._openai_video_headers()
        headers["Accept"] = "application/binary"
        endpoints = [
            f"{OPENAI_API_BASE}/videos/{job_id}/content",
            f"{OPENAI_API_BASE}/videos/generations/{job_id}/content",
        ]

        last_error = ""
        for endpoint in endpoints:
            response = requests.get(endpoint, headers=headers, timeout=240)
            if response.status_code in {404, 405, 501}:
                last_error = response.text[:400]
                continue
            if not response.ok:
                raise RuntimeError(
                    f"OpenAI Sora content download failed: {response.status_code} {response.text[:500]}"
                )

            self.download_dir.mkdir(parents=True, exist_ok=True)
            variant_index = int(re.sub(r"[^0-9]", "", str(suffix)) or "0")
            file_path = self.download_dir / self._compose_output_filename("video", variant_index, "mp4")
            file_path.write_bytes(response.content)
            if file_path.stat().st_size <= 0:
                raise RuntimeError("OpenAI Sora content endpoint returned an empty file.")
            return file_path

        raise RuntimeError(
            "OpenAI Sora completed but no direct URL or downloadable content endpoint was available. "
            f"Last error: {last_error or 'none'}"
        )


    def _seedance_headers(self) -> dict[str, str]:
        api_key = str(self.prompt_config.seedance_settings.get("api_key") or "").strip()
        oauth_token = str(self.prompt_config.seedance_settings.get("oauth_token") or "").strip()
        credential = api_key or oauth_token
        if not credential:
            raise RuntimeError("Seedance API key or OAuth token is required.")
        return {"Authorization": f"Bearer {credential}", "Content-Type": "application/json"}

    def _seedance_payload_variants(self, prompt: str) -> list[dict[str, object]]:
        settings = self.prompt_config.seedance_settings if isinstance(self.prompt_config.seedance_settings, dict) else {}
        model = str(settings.get("model") or "seedance-2.0")
        aspect_ratio = str(settings.get("aspect_ratio") or "16:9")
        resolution = str(settings.get("resolution") or self.prompt_config.video_resolution or "1280x720")
        duration = int(settings.get("duration_seconds") or self.prompt_config.video_duration_seconds or 8)
        fps = int(settings.get("fps") or 24)
        motion_strength = float(settings.get("motion_strength") or 0.6)
        guidance_scale = float(settings.get("guidance_scale") or 7.5)
        seed_value = str(settings.get("seed") or "").strip()
        negative_prompt = str(settings.get("negative_prompt") or "").strip()
        camera_motion = str(settings.get("camera_motion") or "").strip()
        style_preset = str(settings.get("style_preset") or "").strip()

        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "duration_seconds": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
            "motion_strength": motion_strength,
            "guidance_scale": guidance_scale,
            "watermark": bool(settings.get("watermark", False)),
        }
        if seed_value:
            try:
                payload["seed"] = int(seed_value)
            except ValueError:
                payload["seed"] = seed_value
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if camera_motion:
            payload["camera_motion"] = camera_motion
        if style_preset:
            payload["style_preset"] = style_preset

        extra_body = settings.get("extra_body") if isinstance(settings.get("extra_body"), dict) else {}
        payload.update(extra_body)

        variant_without_duration = dict(payload)
        variant_without_duration.pop("duration_seconds", None)
        return [payload, variant_without_duration]

    def _start_seedance_video_job(self, prompt: str) -> tuple[str | None, str | None]:
        self._ensure_not_stopped()
        headers = self._seedance_headers()
        endpoints = [
            f"{SEEDANCE_API_BASE}/videos/generations",
            f"{SEEDANCE_API_BASE}/videos",
            f"{SEEDANCE_API_BASE}/video/generations",
        ]
        last_error = ""
        for endpoint in endpoints:
            for payload in self._seedance_payload_variants(prompt):
                response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
                if response.ok:
                    data = response.json() if response.content else {}
                    job_id = data.get("id") or data.get("job_id")
                    direct_url = data.get("video_url") or data.get("url") or (data.get("output") or {}).get("video_url")
                    return job_id, direct_url
                if response.status_code in {404, 405, 501}:
                    last_error = response.text[:400]
                    break
                body = response.text[:500]
                lower = body.lower()
                if response.status_code in {400, 422} and "duration" in lower:
                    last_error = body
                    continue
                raise RuntimeError(f"Seedance request failed: {response.status_code} {body}")

        raise RuntimeError(
            "Seedance API endpoint not available for this account/base URL. "
            f"Last error: {last_error or 'none'}"
        )

    def _poll_seedance_video_job(self, job_id: str, timeout_s: int = 900) -> dict:
        self._ensure_not_stopped()
        headers = self._seedance_headers()
        endpoints = [
            f"{SEEDANCE_API_BASE}/videos/generations/{job_id}",
            f"{SEEDANCE_API_BASE}/videos/{job_id}",
            f"{SEEDANCE_API_BASE}/video/generations/{job_id}",
        ]
        start = time.time()
        while time.time() - start < timeout_s:
            self._ensure_not_stopped()
            for endpoint in endpoints:
                response = requests.get(endpoint, headers=headers, timeout=60)
                if response.status_code in {404, 405, 501}:
                    continue
                if not response.ok:
                    raise RuntimeError(f"Seedance poll failed: {response.status_code} {response.text[:500]}")
                payload = response.json() if response.content else {}
                status = str(payload.get("status") or payload.get("state") or "").lower()
                if status in {"succeeded", "completed", "ready"}:
                    return payload
                if status in {"failed", "error", "cancelled", "canceled"}:
                    raise RuntimeError(f"Seedance generation failed: {payload.get('error') or payload.get('last_error') or 'unknown error'}")
            time.sleep(5)

        raise TimeoutError("Timed out waiting for Seedance video generation")

    def _resolution_label_for_value(self, resolution: str) -> str:
        mapping = {
            "854x480": "480p",
            "1280x720": "720p",
        }
        return mapping.get(str(resolution), str(resolution))

    def _start_video_job_with_resolution_fallback(self, prompt: str, duration_seconds: int) -> tuple[str, str, str]:
        requested_resolution = str(self.prompt_config.video_resolution)
        requested_label = str(self.prompt_config.video_resolution_label)

        try:
            job_id = self.start_video_job(prompt, requested_resolution, duration_seconds)
            return job_id, requested_resolution, requested_label
        except requests.HTTPError as exc:
            fallback_resolution = "854x480"
            should_try_fallback = requested_resolution == "1280x720" and fallback_resolution != requested_resolution
            if not should_try_fallback:
                raise RuntimeError(
                    "Could not start a video generation job with the selected resolution "
                    f"({requested_label})."
                ) from exc

            fallback_label = self._resolution_label_for_value(fallback_resolution)
            self.status.emit(
                f"Requested resolution {requested_label} is unavailable for this prompt/model; retrying with {fallback_label}."
            )
            try:
                job_id = self.start_video_job(prompt, fallback_resolution, duration_seconds)
                return job_id, fallback_resolution, fallback_label
            except requests.HTTPError as fallback_exc:
                raise RuntimeError(
                    "Could not start a video generation job with the selected resolution "
                    f"({requested_label}) or fallback ({fallback_label})."
                ) from fallback_exc

    def poll_video_job(self, job_id: str, timeout_s: int = 420) -> dict:
        start = time.time()
        while time.time() - start < timeout_s:
            self._ensure_not_stopped()
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
        self._ensure_not_stopped()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        variant_index = int(re.sub(r"[^0-9]", "", str(suffix)) or "0")
        file_path = self.download_dir / self._compose_output_filename("video", variant_index, "mp4")
        with requests.get(video_url, stream=True, timeout=240) as response:
            response.raise_for_status()
            with open(file_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    self._ensure_not_stopped()
                    if chunk:
                        handle.write(chunk)
        return file_path

    def generate_one_video(self, variant: int) -> dict:
        prompt = self.build_prompt(variant)

        selected_duration = int(self.prompt_config.video_duration_seconds)

        if self.prompt_config.video_provider == "openai":
            self.status.emit("Starting OpenAI Sora video generation job...")
            openai_job_id, direct_video_url, initial_reference_id = self._start_openai_video_job(prompt)
            self._openai_last_generated_reference_id = initial_reference_id
            if direct_video_url:
                video_url = direct_video_url
            elif openai_job_id:
                result = self._poll_openai_video_job(openai_job_id)
                video_url = self._extract_video_url_from_payload(result)
                polled_reference_id = self._extract_openai_input_reference_id(result)
                if polled_reference_id:
                    self._openai_last_generated_reference_id = polled_reference_id
            else:
                raise RuntimeError("OpenAI Sora did not return a job id or downloadable video URL.")

            downloaded_from_content_endpoint = False
            if not video_url and openai_job_id:
                self.status.emit("OpenAI Sora returned no direct URL; downloading from content endpoint...")
                file_path = self._download_openai_video_content(openai_job_id, f"v{variant}")
                downloaded_from_content_endpoint = True

            if not video_url and not openai_job_id:
                raise RuntimeError("OpenAI Sora completed but no downloadable video URL or job id was returned.")

            effective_resolution = self.prompt_config.video_resolution
            effective_resolution_label = self.prompt_config.video_resolution_label
        elif self.prompt_config.video_provider == "seedance":
            self.status.emit("Starting Seedance 2.0 video generation job...")
            seedance_job_id, direct_video_url = self._start_seedance_video_job(prompt)
            if direct_video_url:
                video_url = direct_video_url
            elif seedance_job_id:
                result = self._poll_seedance_video_job(seedance_job_id)
                video_url = self._extract_video_url_from_payload(result)
            else:
                raise RuntimeError("Seedance did not return a job id or downloadable video URL.")
            if not video_url:
                raise RuntimeError("Seedance completed but no downloadable video URL was returned.")
            effective_resolution = self.prompt_config.video_resolution
            effective_resolution_label = self.prompt_config.video_resolution_label
        else:
            video_job_id, effective_resolution, effective_resolution_label = self._start_video_job_with_resolution_fallback(
                prompt,
                selected_duration,
            )

            result = self.poll_video_job(video_job_id)
            video_url = result.get("output", {}).get("video_url") or result.get("video_url")
            if not video_url:
                raise RuntimeError("No video URL returned")

        if self.prompt_config.video_provider == "openai" and 'downloaded_from_content_endpoint' in locals() and downloaded_from_content_endpoint:
            resolved_video_url = f"{OPENAI_API_BASE}/videos/{openai_job_id}/content"
        else:
            file_path = self.download_video(video_url, f"v{variant}")
            resolved_video_url = video_url

        if self.prompt_config.video_provider == "openai":
            self._openai_last_generated_video_path = file_path

        return {
            "title": f"Generated Video {variant}",
            "prompt": prompt,
            "resolution": f"{effective_resolution_label} ({self.prompt_config.video_aspect_ratio})",
            "video_file_path": str(file_path),
            "source_url": resolved_video_url,
            "requested_resolution": self.prompt_config.video_resolution,
            "effective_resolution": effective_resolution,
        }

class StitchWorker(QThread):
    progress = Signal(int, str)
    status = Signal(str)
    finished_stitch = Signal(dict)
    failed = Signal(str, str)

    def __init__(
        self,
        window: "MainWindow",
        video_paths: list[Path],
        output_file: Path,
        stitched_base_file: Path,
        crossfade_enabled: bool,
        crossfade_duration: float,
        interpolate_enabled: bool,
        interpolation_fps: int,
        upscale_enabled: bool,
        upscale_target: str,
        use_gpu_encoding: bool,
        custom_music_file: Path | None,
        mute_original_audio: bool,
        original_audio_volume: float,
        music_volume: float,
        audio_fade_duration: float,
    ):
        super().__init__()
        self.window = window
        self.video_paths = video_paths
        self.output_file = output_file
        self.stitched_base_file = stitched_base_file
        self.crossfade_enabled = crossfade_enabled
        self.crossfade_duration = crossfade_duration
        self.interpolate_enabled = interpolate_enabled
        self.interpolation_fps = interpolation_fps
        self.upscale_enabled = upscale_enabled
        self.upscale_target = upscale_target
        self.use_gpu_encoding = use_gpu_encoding
        self.custom_music_file = custom_music_file
        self.mute_original_audio = mute_original_audio
        self.original_audio_volume = original_audio_volume
        self.music_volume = music_volume
        self.audio_fade_duration = audio_fade_duration

    def request_stop(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        enhancement_enabled = self.interpolate_enabled or self.upscale_enabled
        try:
            stitch_target = self.stitched_base_file if enhancement_enabled else self.output_file

            if self.crossfade_enabled:
                self.status.emit(f"Stitching videos with {self.crossfade_duration:.1f}s crossfade transitions enabled.")
                self.window._stitch_videos_with_crossfade(
                    self.video_paths,
                    stitch_target,
                    crossfade_duration=self.crossfade_duration,
                    progress_callback=lambda p: self.progress.emit(max(5, int(5 + (p * 0.70))), "Stitching clips with crossfade..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )
            else:
                self.status.emit("Stitching videos with hard cuts (no crossfade).")
                self.window._stitch_videos_concat(
                    self.video_paths,
                    stitch_target,
                    progress_callback=lambda p: self.progress.emit(max(5, int(5 + (p * 0.70))), "Stitching clips with hard cuts..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            if enhancement_enabled:
                interpolation_status = f"{self.interpolation_fps} fps" if self.interpolate_enabled else "off"
                self.status.emit(
                    "Applying stitched video enhancements: "
                    f"frame interpolation={interpolation_status}, "
                    f"upscaling={self.upscale_target if self.upscale_enabled else 'off'}."
                )
                self.window._enhance_stitched_video(
                    input_file=stitch_target,
                    output_file=self.output_file,
                    interpolate=self.interpolate_enabled,
                    interpolation_fps=self.interpolation_fps,
                    upscale=self.upscale_enabled,
                    upscale_target=self.upscale_target,
                    progress_callback=lambda p: self.progress.emit(max(75, int(75 + (p * 0.25))), "Applying interpolation/upscaling..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            if self.custom_music_file is not None and self.custom_music_file.exists():
                self.status.emit(
                    "Adding custom music track with "
                    f"video audio {'muted' if self.mute_original_audio else f'at {self.original_audio_volume:.0f}%'} "
                    f"and fade in/out {self.audio_fade_duration:.1f}s."
                )
                self.window._apply_custom_music_track(
                    input_file=self.output_file,
                    output_file=self.output_file,
                    music_file=self.custom_music_file,
                    mute_original_audio=self.mute_original_audio,
                    original_audio_volume=self.original_audio_volume,
                    music_volume=self.music_volume,
                    fade_duration=self.audio_fade_duration,
                    progress_callback=lambda p: self.progress.emit(max(92, int(92 + (p * 0.08))), "Mixing custom music..."),
                    use_gpu_encoding=self.use_gpu_encoding,
                )

            self.progress.emit(100, "Finalizing stitched video...")
            self.finished_stitch.emit(
                {
                    "title": "Stitched Video",
                    "prompt": "stitched",
                    "resolution": "mixed",
                    "video_file_path": str(self.output_file),
                    "source_url": "local-stitch",
                }
            )
        except FileNotFoundError:
            self.failed.emit("ffmpeg Missing", "ffmpeg is required for stitching but was not found in PATH.")
        except subprocess.CalledProcessError as exc:
            self.failed.emit("Stitch Failed", exc.stderr[-800:] or "ffmpeg failed.")
        except RuntimeError as exc:
            message = str(exc)
            if "stopped by user" in message.lower():
                self.failed.emit("Stitch Stopped", message)
            else:
                self.failed.emit("Stitch Failed", message)
        finally:
            if self.stitched_base_file.exists():
                self.stitched_base_file.unlink()



class UploadWorker(QThread):
    progress = Signal(int, str)
    finished_upload = Signal(str, str)
    failed = Signal(str, str)

    def __init__(self, platform_name: str, upload_fn: Callable, upload_kwargs: dict):
        super().__init__()
        self.platform_name = platform_name
        self.upload_fn = upload_fn
        self.upload_kwargs = upload_kwargs

    def run(self) -> None:
        try:
            result_id = self.upload_fn(
                **self.upload_kwargs,
                progress_callback=lambda pct, msg: self.progress.emit(max(0, min(100, int(pct))), msg),
            )
            self.finished_upload.emit(self.platform_name, str(result_id))
        except TypeError:
            try:
                result_id = self.upload_fn(**self.upload_kwargs)
            except Exception as exc:
                self.failed.emit(self.platform_name, str(exc))
                return
            self.progress.emit(100, f"{self.platform_name} upload complete.")
            self.finished_upload.emit(self.platform_name, str(result_id))
        except Exception as exc:
            self.failed.emit(self.platform_name, str(exc))


class VideoOverlayWorker(QThread):
    progress = Signal(str)
    finished_overlay = Signal(dict)
    failed = Signal(str, str)

    def __init__(
        self,
        input_video: Path,
        output_video: Path,
        interval_seconds: int,
        subtitle_duration_seconds: float,
        overlay_mode: str,
        manual_text: str,
        font_name: str,
        font_size: int,
        text_position: str,
        ai_callback: Callable[[Path, float], str],
        ai_prompt_callback: Callable[[str, float], str],
        ai_source: str,
    ):
        super().__init__()
        self.input_video = input_video
        self.output_video = output_video
        self.interval_seconds = max(1, int(interval_seconds))
        self.subtitle_duration_seconds = max(0.8, float(subtitle_duration_seconds))
        self.overlay_mode = overlay_mode
        self.manual_text = manual_text.strip()
        self.font_name = self._sanitize_font_name(font_name)
        self.font_size = max(8, min(120, int(font_size)))
        self.ass_alignment = self._position_to_ass_alignment(text_position)
        self.ai_callback = ai_callback
        self.ai_prompt_callback = ai_prompt_callback
        self.ai_source = ai_source


    def _sanitize_font_name(self, value: str) -> str:
        cleaned = re.sub(r"[\r\n]+", " ", value or "").strip()
        cleaned = cleaned.replace(",", " ").replace(":", " ").replace("'", "")
        return cleaned or "Arial"

    def _position_to_ass_alignment(self, value: str) -> int:
        normalized = str(value or "bottom").strip().lower()
        if normalized == "top":
            return 8
        if normalized in {"middle", "center"}:
            return 5
        return 2

    def _ffmpeg_filter_escape(self, value: str) -> str:
        escaped = (value or "").replace("\\", "/")
        escaped = escaped.replace("'", r"\'")
        escaped = escaped.replace(":", r"\:")
        escaped = escaped.replace(",", r"\,")
        return escaped

    def _build_subtitles_filter(self, srt_path: Path) -> str:
        escaped_path = self._ffmpeg_filter_escape(str(srt_path))
        style = (
            f"Alignment={self.ass_alignment},"
            f"FontName={self.font_name},"
            f"FontSize={self.font_size},"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00202020,"
            "BorderStyle=1,"
            "Outline=2,"
            "Shadow=0,"
            "MarginV=24"
        )
        return f"subtitles=filename='{escaped_path}':force_style='{style}'"

    def _extract_frame_image(self, timestamp_seconds: float, frame_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp_seconds:.3f}",
                "-i",
                str(self.input_video),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "png",
                str(frame_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _estimate_frame_brightness(self, frame_path: Path) -> float:
        image = QImage(str(frame_path))
        if image.isNull():
            return 255.0
        sample_w = max(1, image.width() // 48)
        sample_h = max(1, image.height() // 48)
        total = 0.0
        count = 0
        for y in range(0, image.height(), sample_h):
            for x in range(0, image.width(), sample_w):
                color = image.pixelColor(x, y)
                total += (0.2126 * color.red()) + (0.7152 * color.green()) + (0.0722 * color.blue())
                count += 1
        return (total / max(1, count)) if count else 255.0

    def _build_overlay_text(self, frame_path: Path, timestamp_seconds: float) -> str:
        if self.overlay_mode == "manual":
            return self.manual_text

        try:
            if self.manual_text:
                response = self.ai_prompt_callback(self.manual_text, timestamp_seconds)
            else:
                response = self.ai_callback(frame_path, timestamp_seconds)
        except Exception as exc:
            self.progress.emit(
                f"AI caption request failed at {timestamp_seconds:.1f}s ({exc}); skipping subtitle at this timestamp."
            )
            return ""

        first_line = re.split(r"[\r\n]+", (response or "").strip(), maxsplit=1)[0].strip()
        if not first_line:
            self.progress.emit(
                f"AI returned empty caption at {timestamp_seconds:.1f}s; skipping subtitle at this timestamp."
            )
            return ""
        return first_line

    def run(self) -> None:
        temp_files: list[Path] = []
        try:
            duration = _probe_video_duration_seconds(self.input_video)
            if duration <= 0.0:
                raise RuntimeError("Could not read video duration.")

            frame_points: list[float] = []
            current = 0.0
            while current < duration:
                frame_points.append(round(current, 3))
                current += self.interval_seconds
            if not frame_points:
                frame_points = [0.0]

            srt_lines: list[str] = []
            caption_idx = 1
            for idx, point in enumerate(frame_points, start=1):
                self.progress.emit(f"Generating overlay text for {point:.1f}s...")
                frame_path = self.output_video.parent / f"overlay_frame_{self.output_video.stem}_{idx}.png"
                temp_files.append(frame_path)
                self._extract_frame_image(point, frame_path)

                brightness = self._estimate_frame_brightness(frame_path)
                if brightness < 20.0 and (point + 1.0) < duration:
                    alternate_frame_path = self.output_video.parent / f"overlay_frame_{self.output_video.stem}_{idx}_alt.png"
                    temp_files.append(alternate_frame_path)
                    self._extract_frame_image(point + 1.0, alternate_frame_path)
                    alternate_brightness = self._estimate_frame_brightness(alternate_frame_path)
                    if alternate_brightness > brightness:
                        frame_path = alternate_frame_path
                        self.progress.emit(
                            f"Frame at {point:.1f}s was very dark; used {point + 1.0:.1f}s alternative for AI captioning."
                        )

                subtitle_text = _escape_srt_text(self._build_overlay_text(frame_path, point))
                if not subtitle_text.strip():
                    continue

                start_time = point
                end_time = min(duration, point + self.subtitle_duration_seconds)
                if end_time <= start_time:
                    end_time = min(duration, start_time + 0.8)
                srt_lines.extend(
                    [
                        str(caption_idx),
                        f"{_seconds_to_srt_time(start_time)} --> {_seconds_to_srt_time(end_time)}",
                        subtitle_text,
                        "",
                    ]
                )
                caption_idx += 1

            if not srt_lines:
                raise RuntimeError("No AI captions were generated. Try adjusting your prompt or provider settings.")

            srt_path = self.output_video.parent / f"overlay_{self.output_video.stem}.srt"
            srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
            temp_files.append(srt_path)

            self.progress.emit("Rendering video with text overlay...")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(self.input_video),
                    "-vf",
                    self._build_subtitles_filter(srt_path),
                    "-c:a",
                    "copy",
                    str(self.output_video),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.finished_overlay.emit(
                {
                    "title": f"Overlay: {self.input_video.stem}",
                    "prompt": f"overlay-{self.overlay_mode}",
                    "resolution": "same",
                    "video_file_path": str(self.output_video),
                    "source_url": "local-overlay",
                    "overlay_mode": self.overlay_mode,
                    "ai_source": self.ai_source,
                }
            )
        except FileNotFoundError:
            self.failed.emit("ffmpeg Missing", "ffmpeg is required for text overlays but was not found in PATH.")
        except subprocess.CalledProcessError as exc:
            self.failed.emit("Overlay Failed", exc.stderr[-1000:] or "ffmpeg failed.")
        except Exception as exc:
            self.failed.emit("Overlay Failed", str(exc))
        finally:
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass


class BrowserTrainingWorker(QThread):
    status = Signal(str)
    finished = Signal(str, str)
    failed = Signal(str)

    def __init__(self, mode: str, payload: dict):
        super().__init__()
        self.mode = mode
        self.payload = payload

    def run(self) -> None:
        try:
            if self.mode == "train":
                self.status.emit("Starting guided browser training window...")
                trace_path = train_browser_flow(
                    start_url=self.payload["start_url"],
                    output_dir=self.payload["output_dir"],
                    timeout_s=self.payload["timeout_s"],
                )
                self.finished.emit("train", str(trace_path))
                return

            if self.mode == "build":
                self.status.emit("Building reusable process from trace via OpenAI...")
                process_path = build_trained_process(
                    trace_path=self.payload["trace_path"],
                    access_token=self.payload["access_token"],
                    model=self.payload["model"],
                )
                self.finished.emit("build", str(process_path))
                return

            if self.mode == "run":
                self.status.emit("Running trained process in browser...")
                report_path = run_trained_process(
                    process_path=self.payload["process_path"],
                    start_url=self.payload["start_url"],
                    output_dir=self.payload["output_dir"],
                    timeout_s=self.payload["timeout_s"],
                )
                self.finished.emit("run", str(report_path))
                return

            raise RuntimeError(f"Unknown training mode: {self.mode}")
        except Exception as exc:
            self.failed.emit(str(exc))


class AutomationRuntimeWorker(QThread):
    log = Signal(str)
    status = Signal(str, str)

    def __init__(self, extension_dir: Path):
        super().__init__()
        self.extension_dir = extension_dir
        self.loop: asyncio.AbstractEventLoop | None = None
        self.bus: ControlBusServer | None = None
        self.chrome_instance: ChromeInstance | None = None
        self.cdp_controller: CDPController | None = None
        self.udp_service: UdpAutomationService | None = None
        self._udp_started = False

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._startup())
        self.loop.run_forever()

    async def _startup(self) -> None:
        self.bus = ControlBusServer()
        await self.bus.start()
        self.log.emit("Control bus listening at ws://127.0.0.1:18792")

    def _run_coro(self, coro):
        if self.loop is None:
            raise RuntimeError("Automation runtime not started")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=30)

    def start_chrome(self) -> ChromeInstance:
        manager = AutomationChromeManager(extension_dir=self.extension_dir)
        self.chrome_instance = manager.launch_or_reuse()
        self.status.emit("chrome", "started")
        self.log.emit(f"Automation Chrome ready on port {self.chrome_instance.port} (pid={self.chrome_instance.pid})")
        return self.chrome_instance

    def connect_cdp(self) -> str:
        if self.chrome_instance is None:
            raise RuntimeError("Start Automation Chrome first")

        async def _connect() -> str:
            self.cdp_controller = await CDPController.connect(self.chrome_instance.ws_endpoint)
            title = await self.cdp_controller.smoke_test()
            return title

        title = self._run_coro(_connect())
        self.status.emit("cdp", "connected")
        self.log.emit(f"CDP connected. Smoke test title: {title}")
        return title

    def dom_ping(self) -> dict[str, Any]:
        if self.bus is None:
            raise RuntimeError("Control bus is not running")
        clients = list(self.bus.clients.keys())
        if not clients:
            raise RuntimeError("Extension is not connected to the local control bus")

        async def _ping() -> dict[str, Any]:
            return await self.bus.send_dom_ping(clients[0])

        result = self._run_coro(_ping())
        self.log.emit(f"DOM ping result: {result}")
        return result


    def ensure_udp_service(self) -> None:
        async def _start_udp() -> None:
            if self.bus is None:
                raise RuntimeError("Control bus is not running")
            if self.udp_service is None:
                self.udp_service = UdpAutomationService(
                    extension_dir=self.extension_dir,
                    bus=self.bus,
                    start_bus=False,
                )
            if not self._udp_started:
                await self.udp_service.start()
                self._udp_started = True

        self._run_coro(_start_udp())
        self.log.emit("UDP automation service listening on udp://127.0.0.1:18793")

    def stop_runtime(self) -> None:
        if self.loop is None:
            return

        async def _shutdown() -> None:
            if self.udp_service is not None and self._udp_started:
                await self.udp_service.stop()
                self._udp_started = False
            if self.cdp_controller is not None:
                await self.cdp_controller.close()
            if self.bus is not None:
                await self.bus.stop()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(timeout=10)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)





class UdpWorkflowWorker(QThread):
    finished_with_result = Signal(str)
    failed = Signal(str)

    def __init__(self, platform_name: str, video_path: str, title: str, caption: str):
        super().__init__()
        self.platform_name = platform_name
        self.video_path = video_path
        self.title = title
        self.caption = caption
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        executor = UdpExecutor(stop_event=self._stop_event)
        try:
            platform = self.platform_name.lower()
            if platform == "youtube":
                result = udp_youtube_workflow.run(executor, self.video_path, self.title, self.caption)
            elif platform == "tiktok":
                result = udp_tiktok_workflow.run(executor, self.video_path, self.caption)
            elif platform == "facebook":
                result = udp_facebook_workflow.run(executor, self.video_path, self.caption, self.title)
            else:
                raise RuntimeError(f"UDP workflow not implemented for {self.platform_name}")
            self.finished_with_result.emit(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            self.failed.emit(str(exc))


class FilteredWebEnginePage(QWebEnginePage):
    """Suppress noisy third-party console warnings from grok.com in the embedded browser."""

    _IGNORED_CONSOLE_PATTERNS = (
        "cdn-cgi/speculation",
        "react-i18next:: useTranslation",
        "Permissions-Policy header: Unrecognized feature: 'pointer-lock'",
        "violates the following Content Security Policy directive",
        "Play failed: [object DOMException]",
        "[Statsig] A networking error occurred during POST request",
        "featureassets.org/v1/initialize",
        "auth-cdn.oaistatic.com/assets/statsig",
        "upgrade-insecure-requests' is ignored when delivered in a report-only policy",
    )

    def __init__(
        self,
        on_console_message,
        profile: QWebEngineProfile | None = None,
        parent=None,
        auto_file_selector: Callable[[str, object, list[str], list[str]], list[str]] | None = None,
    ):
        if profile is not None:
            super().__init__(profile, parent)
        else:
            super().__init__(parent)
        self._on_console_message = on_console_message
        self._auto_file_selector = auto_file_selector


    def chooseFiles(self, mode, old_files, accepted_mime_types):  # type: ignore[override]
        if self._auto_file_selector:
            try:
                selected = self._auto_file_selector(
                    self.url().toString(),
                    mode,
                    list(old_files or []),
                    list(accepted_mime_types or []),
                )
                if selected:
                    return [str(path) for path in selected if str(path).strip()]
            except Exception:
                pass
        return super().chooseFiles(mode, old_files, accepted_mime_types)

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):  # type: ignore[override]
        if any(pattern in message for pattern in self._IGNORED_CONSOLE_PATTERNS):
            return

        source = str(source_id or "")
        if (
            "Uncaught SyntaxError: Invalid or unexpected token" in str(message)
            and any(
                host in source
                for host in (
                    "instagram.com/create/reel",
                    "facebook.com/reels/create",
                    "facebook.com/reel/",
                    "tiktok.com/tiktokstudio/upload",
                    "tiktok.com/upload",
                )
            )
        ):
            # Social sites sometimes emit their own transient parser errors from first-party scripts.
            # These are noisy and not actionable for our automation flow.
            return

        if self._on_console_message:
            self._on_console_message(f"Browser JS: {message} (source={source_id}:{line_number})")

        super().javaScriptConsoleMessage(level, message, line_number, source_id)


class MainWindow(QMainWindow):
    MANUAL_IMAGE_PICK_RETRY_LIMIT = 3
    MANUAL_IMAGE_SUBMIT_RETRY_LIMIT = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grok Video Desktop Studio")
        self.resize(1500, 900)
        self.download_dir = DOWNLOAD_DIR
        self.videos: list[dict] = []
        self.worker: GenerateWorker | None = None
        self.stitch_worker: StitchWorker | None = None
        self.upload_worker: UploadWorker | None = None
        self.social_upload_pending: dict[str, dict[str, Any]] = {}
        self._cdp_relay_temporarily_disabled = False
        self._cdp_relay_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cdp-relay")
        self.social_upload_timers: dict[str, QTimer] = {}
        self.social_upload_browsers: dict[str, QWebEngineView] = {}
        self.social_upload_status_labels: dict[str, QLabel] = {}
        self.social_upload_progress_bars: dict[str, QProgressBar] = {}
        self.social_upload_tab_indices: dict[str, int] = {}
        self.browser_training_worker: BrowserTrainingWorker | None = None
        self.overlay_worker: VideoOverlayWorker | None = None
        self.automation_runtime: AutomationRuntimeWorker | None = None
        self.automation_chrome_instance: ChromeInstance | None = None
        self.udp_workflow_worker: UdpWorkflowWorker | None = None
        self.embedded_training_active = False
        self.embedded_training_events: list[dict] = []
        self.embedded_training_started_at = 0.0
        self.embedded_training_event_counter = 0
        self.embedded_training_output_dir: Path | None = None
        self.embedded_training_poll_timer = QTimer(self)
        self.embedded_training_poll_timer.setInterval(1000)
        self.embedded_training_poll_timer.timeout.connect(self._poll_embedded_training_events)
        self._ffmpeg_nvenc_checked = False
        self._ffmpeg_nvenc_available = False
        self.preview_fullscreen_overlay_btn: QPushButton | None = None
        self.preview_fullscreen_progress_bar: QProgressBar | None = None
        self.stop_all_requested = False
        self._active_ffmpeg_process: subprocess.Popen[str] | None = None
        self.manual_generation_queue: list[dict] = []
        self.manual_image_generation_queue: list[dict] = []
        self.automation_counter_total = 0
        self.automation_counter_completed = 0
        self.pending_manual_variant_for_download: int | None = None
        self.pending_manual_download_type: str | None = None
        self.pending_manual_image_prompt: str | None = None
        self.pending_manual_redirect_target = "grok"
        self.manual_image_pick_clicked = False
        self.manual_image_video_mode_selected = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_video_mode_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token = 0
        self.manual_download_deadline: float | None = None
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = True
        self.manual_download_in_progress = False
        self.manual_download_started_at: float | None = None
        self.manual_download_poll_timer = QTimer(self)
        self.manual_download_poll_timer.setSingleShot(True)
        self.manual_download_poll_timer.timeout.connect(self._poll_for_manual_video)
        self.continue_from_frame_active = False
        self.continue_from_frame_target_count = 0
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = ""
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path: Path | None = None
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer = QTimer(self)
        self.continue_from_frame_reload_timeout_timer.setSingleShot(True)
        self.continue_from_frame_reload_timeout_timer.timeout.connect(self._on_continue_reload_timeout)
        self.video_playback_hack_timer = QTimer(self)
        self.video_playback_hack_timer.setInterval(1800)
        self.video_playback_hack_timer.setSingleShot(True)
        self.video_playback_hack_timer.timeout.connect(self._ensure_browser_video_playback)
        self._playback_hack_success_logged = False
        self.last_extracted_frame_path: Path | None = None
        self.preview_muted = False
        self.preview_volume = 100
        self._openai_sora_help_always_show = True
        self.custom_music_file: Path | None = None
        self.last_update_prompt_ts = 0
        self.cdp_enabled = False
        self.browser_tab_enabled = {
            "Grok": True,
            "Sora": True,
            "Facebook": True,
            "Instagram": True,
            "TikTok": True,
            "YouTube": True,
            "Sora2Settings": True,
            "SeedanceSettings": True,
            "AIFlowTrainer": True,
        }
        self.ai_social_metadata = AISocialMetadata(
            title="AI Generated Video",
            medium_title="AI Generated Video Clip",
            tiktok_subheading="Swipe for more AI visuals.",
            description="",
            hashtags=["grok", "ai", "generated-video"],
            category="22",
        )
        self._build_ui()
        self._applying_preferences = False
        self._last_saved_preferences_signature: str | None = None
        self._initialize_preferences_autosave()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._load_startup_preferences()
        self._apply_default_theme()
        QTimer.singleShot(1200, self.check_for_updates_on_startup)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "preview", None) and event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
            if self.preview.isFullScreen():
                self._position_preview_fullscreen_overlay()
                self._position_preview_fullscreen_progress_bar()

        if (
            self.stop_all_requested
            and isinstance(watched, QPushButton)
            and watched is not self.stop_all_btn
            and event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress)
        ):
            self._clear_stop_all_flag(watched.text())
        return super().eventFilter(watched, event)

    def _clear_stop_all_flag(self, button_label: str) -> None:
        if not self.stop_all_requested:
            return
        self.stop_all_requested = False
        self._append_log(f"Stop-all flag cleared by button click: {button_label or 'Unnamed button'}")

    def _apply_default_theme(self) -> None:
        self.setStyleSheet("")

    def _instagram_reels_create_url(self) -> str:
        username = ""
        if hasattr(self, "instagram_username") and self.instagram_username is not None:
            username = self.instagram_username.text().strip()
        if not username:
            username = os.getenv("INSTAGRAM_USERNAME", "").strip()
        if username:
            return f"https://www.instagram.com/{username}/reels/create"
        return "https://www.instagram.com/create/reel/"

    def _refresh_instagram_upload_tab_url(self) -> None:
        instagram_browser = self.social_upload_browsers.get("Instagram")
        if instagram_browser is not None:
            instagram_browser.setUrl(QUrl(self._instagram_reels_create_url()))

    def _refresh_facebook_upload_tab_url(self) -> None:
        facebook_browser = self.social_upload_browsers.get("Facebook")
        if facebook_browser is not None:
            facebook_browser.setUrl(QUrl(self._facebook_upload_home_url()))

    def _build_ui(self) -> None:
        splitter = QSplitter()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._build_model_api_settings_dialog()
        self._build_menu_bar()

        self.automation_group = QGroupBox("🤖 Automation Chrome + CDP")
        automation_layout = QVBoxLayout(self.automation_group)
        automation_buttons = QHBoxLayout()

        self.start_automation_chrome_btn = QPushButton("Start Automation Chrome")
        self.start_automation_chrome_btn.clicked.connect(self._start_automation_chrome)
        automation_buttons.addWidget(self.start_automation_chrome_btn)

        self.connect_cdp_btn = QPushButton("Connect CDP")
        self.connect_cdp_btn.clicked.connect(self._connect_automation_cdp)
        automation_buttons.addWidget(self.connect_cdp_btn)

        self.extension_ping_btn = QPushButton("Extension DOM Ping")
        self.extension_ping_btn.clicked.connect(self._run_extension_dom_ping)
        automation_buttons.addWidget(self.extension_ping_btn)

        self.automation_mode = QComboBox()
        self.automation_mode.addItem("Embedded", "embedded")
        self.automation_mode.addItem("UDP", "udp")
        self.automation_mode.setCurrentIndex(1)
        automation_buttons.addWidget(self.automation_mode)

        automation_layout.addLayout(automation_buttons)
        self.automation_log = QTextEdit()
        self.automation_log.setReadOnly(True)
        self.automation_log.setMinimumHeight(100)
        automation_layout.addWidget(self.automation_log)
        left_layout.addWidget(self.automation_group)

        prompt_group = QGroupBox("✨ Prompt Inputs")
        prompt_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        prompt_group_layout = QVBoxLayout(prompt_group)
        prompt_group_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        prompt_group_layout.addWidget(QLabel("Concept"))
        self.concept = QPlainTextEdit()
        self.concept.setPlaceholderText("Describe the video idea...")
        self.concept.setMaximumHeight(90)
        prompt_group_layout.addWidget(self.concept)

        prompt_group_layout.addWidget(QLabel("Manual Prompt (used only when source is Manual)"))
        self.manual_prompt = QPlainTextEdit()
        self.manual_prompt.setPlaceholderText("Paste or write an exact prompt to skip prompt APIs...")
        self.manual_prompt.setPlainText(self.manual_prompt_default_input.toPlainText().strip() or DEFAULT_MANUAL_PROMPT_TEXT)
        self.manual_prompt.setMaximumHeight(110)
        prompt_group_layout.addWidget(self.manual_prompt)

        self.generate_prompt_btn = QPushButton("✨ Generate Prompt + Social Metadata from Concept")
        self.generate_prompt_btn.setToolTip(
            "Uses Prompt Source (Grok/OpenAI) to convert Concept into a 10-second video prompt and social metadata."
        )
        self.generate_prompt_btn.clicked.connect(self.generate_prompt_from_concept)
        prompt_group_layout.addWidget(self.generate_prompt_btn)

        self.count = QSpinBox()
        self.count.setRange(1, 10)
        self.count.setValue(1)

        left_layout.addWidget(prompt_group)

        self.generate_image_btn = QPushButton("🎬 Create New Video")
        self.generate_image_btn.setToolTip("Build and paste an image prompt into the Grok browser tab.")
        self.generate_image_btn.setCheckable(True)
        self.generate_image_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.generate_image_btn, self.start_image_generation)
        )

        self.stop_all_btn = QPushButton("🛑 Stop All Jobs")
        self.stop_all_btn.setToolTip("Stop active generation jobs after current requests complete.")
        self.stop_all_btn.setCheckable(True)
        self.stop_all_btn.clicked.connect(lambda: self._run_with_button_feedback(self.stop_all_btn, self.stop_all_jobs))
        self.stop_all_btn.setMaximumWidth(140)

        self.continue_frame_btn = QPushButton("🟨 Continue Last Video")
        self.continue_frame_btn.setToolTip("Use the last generated video's final frame and continue from it.")
        self.continue_frame_btn.setCheckable(True)
        self.continue_frame_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.continue_frame_btn, self.continue_from_last_frame)
        )

        self.continue_image_btn = QPushButton("🖼️ Create From Image")
        self.continue_image_btn.setToolTip("Choose a local image and continue generation from that frame.")
        self.continue_image_btn.setCheckable(True)
        self.continue_image_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.continue_image_btn, self.continue_from_local_image)
        )

        self.browser_home_btn = QPushButton("🏠 Homepage")
        self.browser_home_btn.setToolTip("Open grok.com/imagine in the embedded browser tab.")
        self.browser_home_btn.setCheckable(True)
        self.browser_home_btn.clicked.connect(lambda: self._run_with_button_feedback(self.browser_home_btn, self.show_browser_page))

        self.sora_generate_image_btn = QPushButton("🎬 Create New Video")
        self.sora_generate_image_btn.setToolTip("Build and paste a video prompt into the Sora browser tab.")
        self.sora_generate_image_btn.setCheckable(True)
        self.sora_generate_image_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.sora_generate_image_btn, self.start_sora_video_generation)
        )

        self.sora_continue_frame_btn = QPushButton("🟨 Continue Last Video")
        self.sora_continue_frame_btn.setToolTip("Use the last generated video's final frame and continue from it.")
        self.sora_continue_frame_btn.setCheckable(True)
        self.sora_continue_frame_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.sora_continue_frame_btn, self.continue_from_last_frame)
        )

        self.sora_continue_image_btn = QPushButton("🖼️ Create From Image")
        self.sora_continue_image_btn.setToolTip("Choose a local image and continue generation from that frame.")
        self.sora_continue_image_btn.setCheckable(True)
        self.sora_continue_image_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.sora_continue_image_btn, self.continue_from_local_image)
        )

        self.sora_browser_home_btn = QPushButton("🏠 Homepage")
        self.sora_browser_home_btn.setToolTip("Open sora.chatgpt.com/drafts in the embedded Sora browser tab.")
        self.sora_browser_home_btn.setCheckable(True)
        self.sora_browser_home_btn.clicked.connect(
            lambda: self._run_with_button_feedback(self.sora_browser_home_btn, self.show_sora_browser_page)
        )

        self.generate_image_btn.setMaximumWidth(170)
        self.continue_frame_btn.setMaximumWidth(170)
        self.continue_image_btn.setMaximumWidth(170)
        self.browser_home_btn.setMaximumWidth(170)
        self.sora_generate_image_btn.setMaximumWidth(170)
        self.sora_continue_frame_btn.setMaximumWidth(170)
        self.sora_continue_image_btn.setMaximumWidth(170)
        self.sora_browser_home_btn.setMaximumWidth(170)

        self.stitch_btn = QPushButton("🧵 Stitch All Videos")
        self.stitch_btn.setToolTip("Combine all downloaded videos into one stitched output file.")
        self.stitch_btn.clicked.connect(self.stitch_all_videos)

        self.stitch_crossfade_checkbox = QCheckBox("Enable 0.5s crossfade between clips")
        self.stitch_crossfade_checkbox.setToolTip("Blend each clip transition using a 0.5 second crossfade.")

        self.stitch_interpolation_checkbox = QCheckBox("Enable frame interpolation")
        self.stitch_interpolation_checkbox.setToolTip(
            "After stitching, use ffmpeg minterpolate to smooth motion by generating in-between frames."
        )

        self.stitch_interpolation_fps = QComboBox()
        self.stitch_interpolation_fps.addItem("48 fps", 48)
        self.stitch_interpolation_fps.addItem("60 fps", 60)
        self.stitch_interpolation_fps.setCurrentIndex(0)
        self.stitch_interpolation_fps.setToolTip("Target frame rate used when frame interpolation is enabled.")

        self.stitch_upscale_checkbox = QCheckBox("Enable AI-style upscaling")
        self.stitch_upscale_checkbox.setToolTip(
            "After stitching, upscale output to a selected target resolution using high-quality Lanczos scaling."
        )

        self.stitch_upscale_target = QComboBox()
        self.stitch_upscale_target.addItem("2x (max 4K)", "2x")
        self.stitch_upscale_target.addItem("1080p (1920x1080)", "1080p")
        self.stitch_upscale_target.addItem("1440p (2560x1440)", "1440p")
        self.stitch_upscale_target.addItem("4K (3840x2160)", "4k")
        self.stitch_upscale_target.setCurrentIndex(0)
        self.stitch_upscale_target.setToolTip("Choose output upscale target resolution.")

        self.stitch_gpu_checkbox = QCheckBox("Use GPU encoding for stitching (NVENC)")
        self.stitch_gpu_checkbox.setToolTip("Use NVIDIA NVENC encoder when available to reduce CPU load.")
        self.stitch_gpu_checkbox.setChecked(True)
        self.stitch_gpu_checkbox.toggled.connect(lambda _: self._sync_video_options_label())

        self.video_options_dropdown = QComboBox()
        self.video_options_dropdown.addItem("0.2s", 0.2)
        self.video_options_dropdown.addItem("0.3s", 0.3)
        self.video_options_dropdown.addItem("0.5s", 0.5)
        self.video_options_dropdown.addItem("0.8s", 0.8)
        self.video_options_dropdown.addItem("1.0s", 1.0)
        self.video_options_dropdown.addItem("Advanced...", None)
        self.video_options_dropdown.setCurrentIndex(2)
        self.video_options_dropdown.setMaximumWidth(140)
        self.video_options_dropdown.setToolTip("Crossfade duration for stitching.")
        self.video_options_dropdown.currentIndexChanged.connect(self._on_video_options_selected)

        self.music_file_label = QLabel("Music: none selected")
        self.music_file_label.setStyleSheet("color: #9fb3c8;")
        self.music_file_label.setWordWrap(True)

        music_actions_layout = QHBoxLayout()
        self.choose_music_btn = QPushButton("🎵 Choose Music (wav/mp3)")
        self.choose_music_btn.setToolTip("Select a local WAV or MP3 file to mix under the stitched video.")
        self.choose_music_btn.clicked.connect(self._choose_custom_music_file)
        music_actions_layout.addWidget(self.choose_music_btn)

        self.clear_music_btn = QPushButton("Clear Music")
        self.clear_music_btn.setToolTip("Remove any selected custom background music file.")
        self.clear_music_btn.clicked.connect(self._clear_custom_music_file)
        music_actions_layout.addWidget(self.clear_music_btn)
        self.music_actions_row = music_actions_layout

        self.stitch_mute_original_checkbox = QCheckBox("Mute original video audio when music is used")
        self.stitch_mute_original_checkbox.setToolTip("If enabled, only the selected music is audible in the stitched output.")

        self.stitch_original_audio_volume = QSpinBox()
        self.stitch_original_audio_volume.setRange(0, 200)
        self.stitch_original_audio_volume.setValue(100)
        self.stitch_original_audio_volume.setPrefix("Original audio: ")
        self.stitch_original_audio_volume.setSuffix("%")
        self.stitch_original_audio_volume.setToolTip("Original video audio level used during custom music mixing.")

        self.stitch_music_volume = QSpinBox()
        self.stitch_music_volume.setRange(0, 200)
        self.stitch_music_volume.setValue(100)
        self.stitch_music_volume.setPrefix("Music audio: ")
        self.stitch_music_volume.setSuffix("%")
        self.stitch_music_volume.setToolTip("Custom music level used during stitched output mixing.")

        self.stitch_audio_fade_duration = QDoubleSpinBox()
        self.stitch_audio_fade_duration.setRange(0.0, 10.0)
        self.stitch_audio_fade_duration.setSingleStep(0.1)
        self.stitch_audio_fade_duration.setDecimals(1)
        self.stitch_audio_fade_duration.setValue(0.5)
        self.stitch_audio_fade_duration.setSuffix(" s")
        self.stitch_audio_fade_duration.setToolTip("Fade-in and fade-out duration applied to stitched output audio mix.")

        self.stitch_audio_fade_label = QLabel("Audio fade in/out")
        self.stitch_audio_fade_label.setStyleSheet("color: #9fb3c8;")

        self.buy_coffee_btn = QPushButton("☕ Buy Me a Coffee")
        self.buy_coffee_btn.setToolTip("If this saves you hours, grab me a ☕")
        self.buy_coffee_btn.clicked.connect(self.open_buy_me_a_coffee)

        generated_videos_group = QGroupBox("🎬 Generated Videos")
        generated_videos_layout = QVBoxLayout(generated_videos_group)
        generated_videos_layout.setContentsMargins(8, 8, 8, 8)
        generated_videos_layout.setSpacing(6)

        generated_videos_layout.addWidget(self.stitch_btn)
        generated_videos_layout.addWidget(self.music_file_label)
        generated_videos_layout.addLayout(self.music_actions_row)

        view_toggle_row = QHBoxLayout()
        view_toggle_row.addStretch(1)
        self.video_view_toggle_btn = QToolButton()
        self.video_view_toggle_btn.setText("☷")
        self.video_view_toggle_btn.setCheckable(True)
        self.video_view_toggle_btn.setToolTip("Toggle between thumbnail grid and details list view.")
        self.video_view_toggle_btn.toggled.connect(self._toggle_video_view_mode)
        view_toggle_row.addWidget(self.video_view_toggle_btn)
        generated_videos_layout.addLayout(view_toggle_row)

        self.video_view_stack = QStackedWidget()
        self.video_grid = QListWidget()
        self.video_grid.setViewMode(QListWidget.IconMode)
        self.video_grid.setResizeMode(QListWidget.Adjust)
        self.video_grid.setMovement(QListWidget.Static)
        self.video_grid.setWrapping(True)
        self.video_grid.setUniformItemSizes(True)
        self.video_grid.setSpacing(10)
        self.video_grid.setIconSize(QPixmap(180, 102).size())
        self.video_grid.setGridSize(QPixmap(220, 138).size())
        self.video_grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.video_grid.setMinimumHeight(354)
        self.video_grid.currentRowChanged.connect(self.show_selected_video)

        self.video_details = QTreeWidget()
        self.video_details.setColumnCount(3)
        self.video_details.setHeaderLabels(["Date", "Filename", "Resolution"])
        self.video_details.setRootIsDecorated(False)
        self.video_details.setSelectionMode(QAbstractItemView.SingleSelection)
        self.video_details.setAlternatingRowColors(True)
        self.video_details.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.video_details.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.video_details.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.video_details.setMinimumHeight(354)
        self.video_details.currentItemChanged.connect(self._on_video_details_selection_changed)

        self.video_view_stack.addWidget(self.video_grid)
        self.video_view_stack.addWidget(self.video_details)
        generated_videos_layout.addWidget(self.video_view_stack)

        video_list_controls = QHBoxLayout()
        self.open_video_btn = QPushButton("📂 Open Video(s)")
        self.open_video_btn.setToolTip("Open one or more local video files and add them to Generated Videos.")
        self.open_video_btn.clicked.connect(self.open_local_video)
        video_list_controls.addWidget(self.open_video_btn)

        self.video_move_up_btn = QPushButton("⬆ Move Up")
        self.video_move_up_btn.setToolTip("Move selected video earlier in the Generated Videos order.")
        self.video_move_up_btn.clicked.connect(lambda: self.move_selected_video(-1))
        video_list_controls.addWidget(self.video_move_up_btn)

        self.video_move_down_btn = QPushButton("⬇ Move Down")
        self.video_move_down_btn.setToolTip("Move selected video later in the Generated Videos order.")
        self.video_move_down_btn.clicked.connect(lambda: self.move_selected_video(1))
        video_list_controls.addWidget(self.video_move_down_btn)

        self.video_remove_btn = QPushButton("🗑 Remove")
        self.video_remove_btn.setToolTip("Remove selected video from Generated Videos list.")
        self.video_remove_btn.clicked.connect(self.remove_selected_video)
        video_list_controls.addWidget(self.video_remove_btn)

        self.video_overlay_btn = QPushButton("📝 Overlay Text")
        self.video_overlay_btn.setToolTip("Create subtitle-style text overlays on the selected video.")
        self.video_overlay_btn.clicked.connect(lambda: self._run_with_button_feedback(self.video_overlay_btn, self.add_overlay_to_selected_video))
        video_list_controls.addWidget(self.video_overlay_btn)

        generated_videos_layout.addLayout(video_list_controls)
        left_layout.addWidget(generated_videos_group)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.preview = QVideoWidget()
        self.preview.installEventFilter(self)
        self.player.setVideoOutput(self.preview)

        self.grok_browser_view = QWebEngineView()
        self.sora_browser = QWebEngineView()
        self.browser = self.grok_browser_view
        if QTWEBENGINE_USE_DISK_CACHE:
            self.browser_profile = QWebEngineProfile("grok-video-desktop-profile", self)
        else:
            # Off-the-record profile avoids startup cache/quota errors on locked folders (common on synced drives).
            self.browser_profile = QWebEngineProfile(self)

        self.grok_browser_view.setPage(FilteredWebEnginePage(self._append_log, self.browser_profile, self.grok_browser_view))
        self.sora_browser.setPage(FilteredWebEnginePage(self._append_log, self.browser_profile, self.sora_browser))
        browser_profile = self.browser_profile
        if QTWEBENGINE_USE_DISK_CACHE:
            (CACHE_DIR / "profile").mkdir(parents=True, exist_ok=True)
            (CACHE_DIR / "cache").mkdir(parents=True, exist_ok=True)
            browser_profile.setPersistentStoragePath(str(CACHE_DIR / "profile"))
            browser_profile.setCachePath(str(CACHE_DIR / "cache"))
            browser_profile.setPersistentCookiesPolicy(QWebEngineProfile.ForcePersistentCookies)
            browser_profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
            browser_profile.setHttpCacheMaximumSize(_env_int("GROK_BROWSER_DISK_CACHE_BYTES", 536870912))
        else:
            browser_profile.setPersistentCookiesPolicy(QWebEngineProfile.NoPersistentCookies)
            browser_profile.setHttpCacheType(QWebEngineProfile.MemoryHttpCache)

        for embedded_browser in (self.grok_browser_view, self.sora_browser):
            browser_settings = embedded_browser.settings()
            browser_settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
            browser_settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            browser_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            browser_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            developer_extras_attr = getattr(QWebEngineSettings.WebAttribute, "DeveloperExtrasEnabled", None)
            if developer_extras_attr is not None:
                browser_settings.setAttribute(developer_extras_attr, True)

        self.grok_browser_view.setUrl(QUrl("https://grok.com/imagine"))
        self.sora_browser.setUrl(QUrl("https://sora.chatgpt.com/drafts"))
        self.grok_browser_view.loadFinished.connect(self._on_browser_load_finished)
        self.sora_browser.loadFinished.connect(self._on_browser_load_finished)
        self.browser_profile.downloadRequested.connect(self._on_browser_download_requested)

        log_group = QGroupBox("📡 Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        log_layout.addWidget(self.log)

        log_actions_layout = QHBoxLayout()
        log_actions_layout.addWidget(self.stop_all_btn, alignment=Qt.AlignLeft)

        self.clear_log_btn = QPushButton("🧹 Clear Log")
        self.clear_log_btn.setToolTip("Clear all activity log entries.")
        self.clear_log_btn.clicked.connect(self.clear_activity_log)
        log_actions_layout.addWidget(self.clear_log_btn, alignment=Qt.AlignLeft)

        self.jump_to_bottom_btn = QPushButton("⤓ Jump to Bottom")
        self.jump_to_bottom_btn.setToolTip("Jump to the latest activity log entry.")
        self.jump_to_bottom_btn.clicked.connect(self.jump_activity_log_to_bottom)
        log_actions_layout.addWidget(self.jump_to_bottom_btn, alignment=Qt.AlignLeft)

        log_actions_layout.addStretch(1)
        log_actions_layout.addWidget(self.buy_coffee_btn, alignment=Qt.AlignRight)
        log_layout.addLayout(log_actions_layout)

        if QTWEBENGINE_USE_DISK_CACHE:
            self._append_log(f"Browser cache path: {CACHE_DIR}")
        else:
            self._append_log(
                "Browser cache: running in memory/off-the-record mode because no writable cache folder was available."
            )

        preview_group = QGroupBox("🎞️ Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview)

        preview_controls = QHBoxLayout()
        self.preview_play_btn = QPushButton("▶️")
        self.preview_play_btn.setToolTip("Play the selected video in the preview pane.")
        self.preview_play_btn.clicked.connect(self.play_preview)
        preview_controls.addWidget(self.preview_play_btn)
        self.preview_stop_btn = QPushButton("⏹️")
        self.preview_stop_btn.setToolTip("Stop playback in the preview pane.")
        self.preview_stop_btn.clicked.connect(self.stop_preview)
        preview_controls.addWidget(self.preview_stop_btn)

        self.preview_mute_checkbox = QCheckBox("🔇")
        self.preview_mute_checkbox.setToolTip("Mute/unmute preview audio.")
        self.preview_mute_checkbox.toggled.connect(self._set_preview_muted)
        preview_controls.addWidget(self.preview_mute_checkbox)

        self.preview_volume_label = QLabel("🔊")
        preview_controls.addWidget(self.preview_volume_label)

        self.preview_volume_slider = QSpinBox()
        self.preview_volume_slider.setRange(0, 100)
        self.preview_volume_slider.setValue(self.preview_volume)
        self.preview_volume_slider.setSuffix("%")
        self.preview_volume_slider.setButtonSymbols(QSpinBox.NoButtons)
        self.preview_volume_slider.setFixedWidth(58)
        self.preview_volume_slider.setStyleSheet("font-size: 11px;")
        self.preview_volume_slider.valueChanged.connect(self._set_preview_volume)
        preview_controls.addWidget(self.preview_volume_slider)

        self.preview_fullscreen_btn = QPushButton("⛶")
        self.preview_fullscreen_btn.setToolTip("Toggle fullscreen preview.")
        self.preview_fullscreen_btn.clicked.connect(self.toggle_preview_fullscreen)
        self.preview.fullScreenChanged.connect(self._on_preview_fullscreen_changed)
        preview_controls.addWidget(self.preview_fullscreen_btn)
        preview_layout.addLayout(preview_controls)

        timeline_layout = QHBoxLayout()
        self.preview_position_label = QLabel("00:00 / 00:00")
        timeline_layout.addWidget(self.preview_position_label)

        self.preview_seek_slider = QSlider(Qt.Horizontal)
        self.preview_seek_slider.setRange(0, 0)
        self.preview_seek_slider.sliderMoved.connect(self.seek_preview)
        timeline_layout.addWidget(self.preview_seek_slider)
        preview_layout.addLayout(timeline_layout)

        self.audio_output.setMuted(self.preview_muted)
        self.audio_output.setVolume(self.preview_volume / 100)
        self.player.positionChanged.connect(self._on_preview_position_changed)
        self.player.durationChanged.connect(self._on_preview_duration_changed)

        bottom_splitter = QSplitter()
        bottom_splitter.setOpaqueResize(True)
        bottom_splitter.setChildrenCollapsible(False)
        bottom_splitter.addWidget(preview_group)
        bottom_splitter.addWidget(log_group)
        bottom_splitter.setSizes([500, 800])

        self.grok_browser_tab = QWidget()
        grok_browser_layout = QVBoxLayout(self.grok_browser_tab)
        grok_browser_controls = QGridLayout()
        grok_browser_controls.addWidget(self.generate_image_btn, 0, 0)
        grok_browser_controls.addWidget(self.continue_frame_btn, 0, 1)
        grok_browser_controls.addWidget(self.continue_image_btn, 0, 2)
        grok_browser_controls.addWidget(self.browser_home_btn, 0, 3)

        self.sora_browser_tab = QWidget()
        sora_browser_layout = QVBoxLayout(self.sora_browser_tab)
        sora_browser_controls = QGridLayout()
        sora_browser_controls.addWidget(self.sora_generate_image_btn, 0, 0)
        sora_browser_controls.addWidget(self.sora_continue_frame_btn, 0, 1)
        sora_browser_controls.addWidget(self.sora_continue_image_btn, 0, 2)
        sora_browser_controls.addWidget(self.sora_browser_home_btn, 0, 3)

        self.video_resolution = QComboBox()
        self.video_resolution.addItem("480p (854x480)", "854x480")
        self.video_resolution.addItem("720p (1280x720)", "1280x720")
        self.video_resolution.setCurrentIndex(1)
        self.video_duration = QComboBox()
        self.video_duration.addItem("6s", 6)
        self.video_duration.addItem("10s", 10)
        self.video_duration.setCurrentIndex(1)
        self.video_aspect_ratio = QComboBox()
        self.video_aspect_ratio.addItem("2:3", "2:3")
        self.video_aspect_ratio.addItem("3:2", "3:2")
        self.video_aspect_ratio.addItem("1:1", "1:1")
        self.video_aspect_ratio.addItem("9:16", "9:16")
        self.video_aspect_ratio.addItem("16:9", "16:9")
        self.video_aspect_ratio.setCurrentIndex(4)
        grok_browser_layout.addLayout(grok_browser_controls)
        grok_browser_layout.addWidget(self.grok_browser_view)

        sora_browser_layout.addLayout(sora_browser_controls)
        sora_browser_layout.addWidget(self.sora_browser)

        self.browser_tabs = QTabWidget()
        self.grok_browser_tab_index = self.browser_tabs.addTab(self.grok_browser_tab, "Grok Browser")
        self.sora_browser_tab_index = self.browser_tabs.addTab(self.sora_browser_tab, "Sora Browser")
        self.social_upload_tab_indices["Facebook"] = self.browser_tabs.addTab(
            self._build_social_upload_tab("Facebook", self._facebook_upload_home_url()),
            "Facebook Upload",
        )
        self.social_upload_tab_indices["Instagram"] = self.browser_tabs.addTab(
            self._build_social_upload_tab("Instagram", self._instagram_reels_create_url()),
            "Instagram Upload",
        )
        self.social_upload_tab_indices["TikTok"] = self.browser_tabs.addTab(
            self._build_social_upload_tab("TikTok", "https://www.tiktok.com/upload"),
            "TikTok Upload",
        )
        self.social_upload_tab_indices["YouTube"] = self.browser_tabs.addTab(
            self._build_social_upload_tab("YouTube", "https://studio.youtube.com"),
            "YouTube Upload",
        )
        self.sora2_settings_tab_index = self.browser_tabs.addTab(self._build_sora2_settings_tab(), "Sora 2 Video Settings")
        self.seedance_settings_tab_index = self.browser_tabs.addTab(self._build_seedance_settings_tab(), "Seedance 2.0 Video Settings")
        self.ai_flow_trainer_tab_index = self.browser_tabs.addTab(self._build_browser_training_tab(), "AI Flow Trainer")
        self.browser_tabs.currentChanged.connect(self._on_browser_tab_changed)

        self.browser_tab_indices = {
            "Grok": self.grok_browser_tab_index,
            "Sora": self.sora_browser_tab_index,
            **self.social_upload_tab_indices,
            "Sora2Settings": self.sora2_settings_tab_index,
            "SeedanceSettings": self.seedance_settings_tab_index,
            "AIFlowTrainer": self.ai_flow_trainer_tab_index,
        }
        self.browser_tab_widgets = {
            "Grok": self.grok_browser_tab,
            "Sora": self.sora_browser_tab,
        }
        for platform_name in self.social_upload_browsers:
            tab_idx = self.social_upload_tab_indices.get(platform_name)
            if tab_idx is not None:
                self.browser_tab_widgets[platform_name] = self.browser_tabs.widget(tab_idx)
        self.browser_tab_webviews = {
            "Grok": self.grok_browser_view,
            "Sora": self.sora_browser,
            **self.social_upload_browsers,
        }

        for tab_key in self.browser_tab_indices:
            self._set_browser_tab_enabled(tab_key, self._is_browser_tab_enabled(tab_key))
        self._refresh_browser_tab_selection()

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setOpaqueResize(True)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self.browser_tabs)
        right_splitter.addWidget(bottom_splitter)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setSizes([620, 280])

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_splitter)
        splitter.setSizes([760, 1140])

        # Keep browser visible as a fixed right-hand pane
        splitter.setChildrenCollapsible(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(False)
        status_bar.setContentsMargins(8, 0, 0, 0)
        self.setStatusBar(status_bar)

        self.upload_progress_label = QLabel("Upload progress: idle")
        self.upload_progress_label.setToolTip("Upload progress details")
        self.upload_progress_label.setVisible(False)
        status_bar.addWidget(self.upload_progress_label, 1)

        self.upload_progress_bar = QProgressBar()
        self.upload_progress_bar.setRange(0, 100)
        self.upload_progress_bar.setValue(0)
        self.upload_progress_bar.setVisible(False)
        self.upload_progress_bar.setFixedWidth(260)
        status_bar.addPermanentWidget(self.upload_progress_bar)

        self.stitch_progress_label = QLabel("Stitch all video progress: idle")
        self.stitch_progress_label.setStyleSheet("color: #9fb3c8;")
        self.stitch_progress_label.setVisible(False)
        status_bar.addWidget(self.stitch_progress_label, 1)

        self.stitch_progress_bar = QProgressBar()
        self.stitch_progress_bar.setRange(0, 100)
        self.stitch_progress_bar.setValue(0)
        self.stitch_progress_bar.setVisible(False)
        self.stitch_progress_bar.setFixedWidth(260)
        status_bar.addPermanentWidget(self.stitch_progress_bar)

        self._populate_top_settings_menus()
        self._toggle_prompt_source_fields()
        self._sync_video_options_label()
        self._reset_cdp_relay_session_state()
        self._refresh_status_bar_visibility()

    def _build_social_upload_tab(self, platform_name: str, upload_url: str) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            f"{platform_name} upload tab: use API upload or browser automation in this dedicated browser so uploads can run in parallel."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(info)

        controls = QHBoxLayout()
        api_btn = QPushButton(f"Upload to {platform_name} via API")
        browser_btn = QPushButton(f"Automate {platform_name} in Browser")
        open_btn = QPushButton("Open Upload Page")

        if platform_name == "Facebook":
            api_btn.clicked.connect(self.upload_selected_to_facebook)
            browser_btn.clicked.connect(self.start_facebook_browser_upload)
        elif platform_name == "Instagram":
            api_btn.clicked.connect(self.upload_selected_to_instagram)
            browser_btn.clicked.connect(self.start_instagram_browser_upload)
        elif platform_name == "TikTok":
            api_btn.clicked.connect(self.upload_selected_to_tiktok)
            browser_btn.clicked.connect(self.start_tiktok_browser_upload)
        else:
            api_btn.clicked.connect(self.upload_selected_to_youtube)
            browser_btn.clicked.connect(self.start_youtube_browser_upload)
            self.upload_youtube_btn = api_btn

        open_btn.clicked.connect(lambda _=False, p=platform_name, u=upload_url: self._open_social_upload_page(p, self._social_upload_url_for_platform(p, u)))
        controls.addWidget(api_btn)
        controls.addWidget(browser_btn)
        controls.addWidget(open_btn)
        layout.addLayout(controls)

        status_label = QLabel("Status: idle")
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setVisible(False)
        layout.addWidget(progress_bar)

        browser = QWebEngineView()
        browser.setPage(
            FilteredWebEnginePage(
                self._append_log,
                self.browser_profile,
                browser,
                auto_file_selector=self._resolve_social_auto_file_selection,
            )
        )
        browser.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        browser.settings().setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        developer_extras_attr = getattr(QWebEngineSettings.WebAttribute, "DeveloperExtrasEnabled", None)
        if developer_extras_attr is not None:
            browser.settings().setAttribute(developer_extras_attr, True)
        browser.setUrl(QUrl(self._social_upload_url_for_platform(platform_name, upload_url)))
        browser.loadFinished.connect(lambda ok, p=platform_name: self._on_social_browser_load_finished(p, ok))
        layout.addWidget(browser, 1)

        self.social_upload_browsers[platform_name] = browser
        self.social_upload_status_labels[platform_name] = status_label
        self.social_upload_progress_bars[platform_name] = progress_bar

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda p=platform_name: self._run_social_browser_upload_step(p))
        self.social_upload_timers[platform_name] = timer

        return tab

    def _facebook_upload_home_url(self) -> str:
        configured = self.facebook_profile_url.text().strip() if hasattr(self, "facebook_profile_url") else ""
        if configured:
            return configured
        return "https://www.facebook.com/"

    def _social_upload_url_for_platform(self, platform_name: str, fallback_url: str) -> str:
        if platform_name == "Facebook":
            return self._facebook_upload_home_url()
        return fallback_url

    def _open_social_upload_page(self, platform_name: str, upload_url: str) -> None:
        if not self._is_browser_tab_enabled(platform_name):
            self._append_log(f"{platform_name} upload tab is disabled. Re-enable it from View → Browser Tabs.")
            return
        browser = self.social_upload_browsers.get(platform_name)
        if browser is None:
            return
        browser.setUrl(QUrl(upload_url))
        self._append_log(f"Opened {platform_name} upload page in dedicated tab.")

    def _on_social_browser_load_finished(self, platform_name: str, ok: bool) -> None:
        if not ok:
            return
        if platform_name in self.social_upload_pending:
            timer = self.social_upload_timers.get(platform_name)
            if timer is not None:
                timer.start(900)

    def _resolve_social_auto_file_selection(
        self,
        source_url: str,
        _mode: object,
        _old_files: list[str],
        _accepted_mime_types: list[str],
    ) -> list[str]:
        url = str(source_url or "").lower()
        platform_map: list[tuple[str, Iterable[str]]] = [
            ("Facebook", ("facebook.com",)),
            ("Instagram", ("instagram.com",)),
            ("TikTok", ("tiktok.com",)),
            ("YouTube", ("youtube.com", "studio.youtube.com")),
        ]

        matched_platform: str | None = None
        for platform_name, patterns in platform_map:
            if any(pattern in url for pattern in patterns):
                matched_platform = platform_name
                break

        candidates: list[str] = []
        if matched_platform:
            candidates.append(matched_platform)
        candidates.extend([name for name in self.social_upload_pending.keys() if name not in candidates])

        for platform_name in candidates:
            pending = self.social_upload_pending.get(platform_name)
            if not pending:
                continue
            video_path = str(pending.get("video_path") or "").strip()
            if not video_path:
                continue
            candidate = Path(video_path)
            if candidate.exists() and candidate.is_file():
                self._append_log(
                    f"{platform_name}: auto-selected file for upload dialog from {url or 'unknown page'}: {candidate.name} ({candidate})"
                )
                return [str(candidate)]
        return []

    def _selected_video_path_for_upload(self) -> str:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            return ""
        return str(self.videos[index].get("video_file_path") or "").strip()

    def _build_sora2_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Configure official Sora 2 create-video parameters used by OpenAI API generation. "
            "These settings are applied when Video Provider is OpenAI Sora 2 API and you click API Generate Video."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(description)

        form = QFormLayout()

        self.sora2_model = QComboBox()
        self.sora2_model.setEditable(True)
        for model_name in [
            "sora-2",
            "sora-2-pro",
            "sora-2-2025-10-06",
            "sora-2-pro-2025-10-06",
            "sora-2-2025-12-08",
        ]:
            self.sora2_model.addItem(model_name, model_name)
        model_env = os.getenv("OPENAI_SORA_MODEL", "sora-2").strip() or "sora-2"
        model_index = self.sora2_model.findData(model_env)
        if model_index >= 0:
            self.sora2_model.setCurrentIndex(model_index)
        else:
            self.sora2_model.setCurrentText(model_env)
        form.addRow("model", self.sora2_model)

        self.sora2_seconds = QComboBox()
        for seconds in ["4", "8", "12"]:
            self.sora2_seconds.addItem(f"{seconds}s", seconds)
        seconds_env = os.getenv("OPENAI_SORA_SECONDS", "8").strip() or "8"
        seconds_index = self.sora2_seconds.findData(seconds_env)
        self.sora2_seconds.setCurrentIndex(seconds_index if seconds_index >= 0 else 1)
        form.addRow("seconds", self.sora2_seconds)

        self.sora2_size = QComboBox()
        for size in ["720x1280", "1280x720", "1024x1792", "1792x1024"]:
            self.sora2_size.addItem(size, size)
        size_env = os.getenv("OPENAI_SORA_SIZE", "1280x720").strip() or "1280x720"
        size_index = self.sora2_size.findData(size_env)
        self.sora2_size.setCurrentIndex(size_index if size_index >= 0 else 1)
        form.addRow("size", self.sora2_size)

        input_ref_row = QHBoxLayout()
        self.sora2_input_reference = QLineEdit(os.getenv("OPENAI_SORA_INPUT_REFERENCE", ""))
        self.sora2_input_reference.setPlaceholderText("Optional file id/reference image for video continuation")
        input_ref_row.addWidget(self.sora2_input_reference)
        pick_input_ref = QPushButton("Browse…")
        pick_input_ref.clicked.connect(self._choose_sora2_input_reference)
        input_ref_row.addWidget(pick_input_ref)
        input_ref_wrap = QWidget()
        input_ref_wrap.setLayout(input_ref_row)
        form.addRow("input_reference (optional)", input_ref_wrap)

        self.sora2_continue_from_last_frame = QCheckBox("Auto-continue from the last generated video frame")
        self.sora2_continue_from_last_frame.setToolTip(
            "When enabled, the app extracts the final frame from the latest generated video, uploads it to OpenAI Files, and sends that file id as input_reference."
        )
        self.sora2_continue_from_last_frame.setChecked(False)
        form.addRow("continuity", self.sora2_continue_from_last_frame)

        self.sora2_extra_body = QPlainTextEdit()
        self.sora2_extra_body.setPlaceholderText('{"any_additional_openai_video_fields": "value"}')
        self.sora2_extra_body.setMaximumHeight(110)
        form.addRow("extra_body JSON (optional)", self.sora2_extra_body)

        layout.addLayout(form)

        hint = QLabel(
            "Known create parameters from current OpenAI SDK docs: prompt, model, seconds, size, input_reference. Continuity mode can automatically use the last generated video frame. "
            "The prompt is generated by the app; fields above control the request payload."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(hint)

        self.sora_generate_btn = QPushButton("🎬 API Generate Video")
        self.sora_generate_btn.setToolTip("Generate and download video via the selected API provider.")
        self.sora_generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.sora_generate_btn)

        layout.addStretch(1)
        return tab

    def _choose_sora2_input_reference(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Reference Media",
            str(self.download_dir),
            "Media Files (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.m4v);;All Files (*)",
        )
        if file_path:
            self.sora2_input_reference.setText(file_path)

    def _collect_sora2_settings(self) -> dict[str, object]:
        settings: dict[str, object] = {
            "model": self.sora2_model.currentData() or self.sora2_model.currentText().strip() or "sora-2",
            "seconds": str(self.sora2_seconds.currentData() or "8"),
            "size": str(self.sora2_size.currentData() or "1280x720"),
            "input_reference": self.sora2_input_reference.text().strip(),
            "continue_from_last_frame": self.sora2_continue_from_last_frame.isChecked(),
            "extra_body": {},
        }

        raw_extra = self.sora2_extra_body.toPlainText().strip()
        if raw_extra:
            parsed = json.loads(raw_extra)
            if not isinstance(parsed, dict):
                raise ValueError("Sora 2 extra_body JSON must be an object.")
            settings["extra_body"] = parsed

        return settings

    def _build_seedance_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Configure Seedance 2.0 video generation options. "
            "These settings are applied when Video Provider is Seedance 2.0 API and you click API Generate Video."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(description)

        form = QFormLayout()

        self.seedance_model = QComboBox()
        self.seedance_model.setEditable(True)
        for model_name in ["seedance-2.0", "seedance-2.0-pro"]:
            self.seedance_model.addItem(model_name, model_name)
        seedance_model_env = os.getenv("SEEDANCE_MODEL", "seedance-2.0").strip() or "seedance-2.0"
        seedance_model_index = self.seedance_model.findData(seedance_model_env)
        if seedance_model_index >= 0:
            self.seedance_model.setCurrentIndex(seedance_model_index)
        else:
            self.seedance_model.setCurrentText(seedance_model_env)
        form.addRow("model", self.seedance_model)

        self.seedance_resolution = QComboBox()
        for size in ["1280x720", "720x1280", "1024x1024", "1920x1080"]:
            self.seedance_resolution.addItem(size, size)
        seedance_size_env = os.getenv("SEEDANCE_RESOLUTION", "1280x720").strip() or "1280x720"
        seedance_size_index = self.seedance_resolution.findData(seedance_size_env)
        self.seedance_resolution.setCurrentIndex(seedance_size_index if seedance_size_index >= 0 else 0)
        form.addRow("resolution", self.seedance_resolution)

        self.seedance_aspect_ratio = QComboBox()
        for ratio in ["16:9", "9:16", "1:1", "4:3", "3:4"]:
            self.seedance_aspect_ratio.addItem(ratio, ratio)
        aspect_env = os.getenv("SEEDANCE_ASPECT_RATIO", "16:9").strip() or "16:9"
        aspect_index = self.seedance_aspect_ratio.findData(aspect_env)
        self.seedance_aspect_ratio.setCurrentIndex(aspect_index if aspect_index >= 0 else 0)
        form.addRow("aspect_ratio", self.seedance_aspect_ratio)

        self.seedance_duration_seconds = QComboBox()
        for seconds in ["4", "6", "8", "10", "12"]:
            self.seedance_duration_seconds.addItem(f"{seconds}s", seconds)
        duration_env = os.getenv("SEEDANCE_DURATION_SECONDS", "8").strip() or "8"
        duration_index = self.seedance_duration_seconds.findData(duration_env)
        self.seedance_duration_seconds.setCurrentIndex(duration_index if duration_index >= 0 else 2)
        form.addRow("duration_seconds", self.seedance_duration_seconds)

        self.seedance_fps = QComboBox()
        for fps in ["24", "30", "60"]:
            self.seedance_fps.addItem(fps, fps)
        fps_env = os.getenv("SEEDANCE_FPS", "24").strip() or "24"
        fps_index = self.seedance_fps.findData(fps_env)
        self.seedance_fps.setCurrentIndex(fps_index if fps_index >= 0 else 0)
        form.addRow("fps", self.seedance_fps)

        self.seedance_motion_strength = QDoubleSpinBox()
        self.seedance_motion_strength.setRange(0.0, 1.0)
        self.seedance_motion_strength.setSingleStep(0.1)
        self.seedance_motion_strength.setDecimals(2)
        self.seedance_motion_strength.setValue(float(os.getenv("SEEDANCE_MOTION_STRENGTH", "0.6") or "0.6"))
        form.addRow("motion_strength", self.seedance_motion_strength)

        self.seedance_guidance_scale = QDoubleSpinBox()
        self.seedance_guidance_scale.setRange(1.0, 20.0)
        self.seedance_guidance_scale.setSingleStep(0.5)
        self.seedance_guidance_scale.setDecimals(1)
        self.seedance_guidance_scale.setValue(float(os.getenv("SEEDANCE_GUIDANCE_SCALE", "7.5") or "7.5"))
        form.addRow("guidance_scale", self.seedance_guidance_scale)

        self.seedance_seed = QLineEdit(os.getenv("SEEDANCE_SEED", ""))
        self.seedance_seed.setPlaceholderText("Optional deterministic seed")
        form.addRow("seed (optional)", self.seedance_seed)

        self.seedance_negative_prompt = QLineEdit(os.getenv("SEEDANCE_NEGATIVE_PROMPT", ""))
        self.seedance_negative_prompt.setPlaceholderText("Optional negative prompt")
        form.addRow("negative_prompt (optional)", self.seedance_negative_prompt)

        self.seedance_camera_motion = QComboBox()
        self.seedance_camera_motion.addItem("Default", "")
        for motion in ["static", "pan_left", "pan_right", "dolly_in", "dolly_out", "orbit"]:
            self.seedance_camera_motion.addItem(motion, motion)
        motion_env = os.getenv("SEEDANCE_CAMERA_MOTION", "").strip()
        motion_index = self.seedance_camera_motion.findData(motion_env)
        self.seedance_camera_motion.setCurrentIndex(motion_index if motion_index >= 0 else 0)
        form.addRow("camera_motion", self.seedance_camera_motion)

        self.seedance_style_preset = QComboBox()
        self.seedance_style_preset.addItem("Default", "")
        for style in ["cinematic", "anime", "photoreal", "3d", "watercolor"]:
            self.seedance_style_preset.addItem(style, style)
        style_env = os.getenv("SEEDANCE_STYLE_PRESET", "").strip()
        style_index = self.seedance_style_preset.findData(style_env)
        self.seedance_style_preset.setCurrentIndex(style_index if style_index >= 0 else 0)
        form.addRow("style_preset", self.seedance_style_preset)

        self.seedance_watermark = QCheckBox("Include watermark")
        self.seedance_watermark.setChecked(os.getenv("SEEDANCE_WATERMARK", "0").strip().lower() in {"1", "true", "yes", "on"})
        form.addRow("watermark", self.seedance_watermark)

        self.seedance_extra_body = QPlainTextEdit()
        self.seedance_extra_body.setPlaceholderText('{"provider_specific_field": "value"}')
        self.seedance_extra_body.setMaximumHeight(110)
        form.addRow("extra_body JSON (optional)", self.seedance_extra_body)

        layout.addLayout(form)

        hint = QLabel(
            "Available options reflected in this tab: model, prompt, duration_seconds, resolution, aspect_ratio, fps, motion_strength, guidance_scale, seed, negative_prompt, camera_motion, style_preset, watermark, and extra_body JSON overrides."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(hint)

        self.seedance_generate_btn = QPushButton("🎬 API Generate Video")
        self.seedance_generate_btn.setToolTip("Generate and download video via the selected API provider.")
        self.seedance_generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.seedance_generate_btn)

        layout.addStretch(1)
        return tab

    def _collect_seedance_settings(self) -> dict[str, object]:
        settings: dict[str, object] = {
            "api_key": self.seedance_api_key.text().strip(),
            "oauth_token": self.seedance_oauth_token.text().strip(),
            "model": self.seedance_model.currentData() or self.seedance_model.currentText().strip() or "seedance-2.0",
            "resolution": str(self.seedance_resolution.currentData() or "1280x720"),
            "aspect_ratio": str(self.seedance_aspect_ratio.currentData() or "16:9"),
            "duration_seconds": int(self.seedance_duration_seconds.currentData() or "8"),
            "fps": int(self.seedance_fps.currentData() or "24"),
            "motion_strength": float(self.seedance_motion_strength.value()),
            "guidance_scale": float(self.seedance_guidance_scale.value()),
            "seed": self.seedance_seed.text().strip(),
            "negative_prompt": self.seedance_negative_prompt.text().strip(),
            "camera_motion": str(self.seedance_camera_motion.currentData() or ""),
            "style_preset": str(self.seedance_style_preset.currentData() or ""),
            "watermark": self.seedance_watermark.isChecked(),
            "extra_body": {},
        }

        raw_extra = self.seedance_extra_body.toPlainText().strip()
        if raw_extra:
            parsed = json.loads(raw_extra)
            if not isinstance(parsed, dict):
                raise ValueError("Seedance extra_body JSON must be an object.")
            settings["extra_body"] = parsed
        return settings

    def _build_browser_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        description = QLabel(
            "Train a reusable browser flow from a guided session, then replay it on demand. "
            "Training opens a separate Playwright browser window."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(description)

        form = QFormLayout()
        self.training_start_url = QLineEdit("https://grok.com/imagine")
        form.addRow("Start URL", self.training_start_url)

        self.training_output_dir = QLineEdit(str(self.download_dir / "browser_training"))
        form.addRow("Output Folder", self.training_output_dir)

        self.training_timeout = QSpinBox()
        self.training_timeout.setRange(30, 7200)
        self.training_timeout.setValue(900)
        self.training_timeout.setSuffix(" s")
        form.addRow("Training Timeout", self.training_timeout)

        self.training_openai_model = QLineEdit("gpt-5.1-codex")
        form.addRow("Planner Model", self.training_openai_model)

        self.training_use_embedded_browser = QCheckBox("Use embedded browser for training")
        self.training_use_embedded_browser.setChecked(True)
        self.training_use_embedded_browser.setToolTip(
            "Record training clicks and inputs directly from the in-app browser tab instead of launching a separate Playwright window."
        )
        form.addRow("Training Capture", self.training_use_embedded_browser)

        self.training_trace_path = QLineEdit()
        self.training_trace_path.setPlaceholderText("Path to raw_training_trace.json")
        form.addRow("Training Trace", self.training_trace_path)

        self.training_process_path = QLineEdit()
        self.training_process_path.setPlaceholderText("Path to *.process.json")
        form.addRow("Process File", self.training_process_path)

        layout.addLayout(form)

        path_actions = QHBoxLayout()
        choose_output_btn = QPushButton("Choose Output Folder")
        choose_output_btn.clicked.connect(self._choose_training_output_folder)
        path_actions.addWidget(choose_output_btn)

        choose_trace_btn = QPushButton("Select Trace")
        choose_trace_btn.clicked.connect(self._choose_training_trace_file)
        path_actions.addWidget(choose_trace_btn)

        choose_process_btn = QPushButton("Select Process")
        choose_process_btn.clicked.connect(self._choose_training_process_file)
        path_actions.addWidget(choose_process_btn)
        layout.addLayout(path_actions)

        run_actions = QHBoxLayout()
        self.training_start_btn = QPushButton("Start Training")
        self.training_start_btn.clicked.connect(self.start_browser_training)
        run_actions.addWidget(self.training_start_btn)

        self.training_stop_btn = QPushButton("Stop Training")
        self.training_stop_btn.setToolTip("Stop embedded training capture and save raw_training_trace.json.")
        self.training_stop_btn.setEnabled(False)
        self.training_stop_btn.clicked.connect(self.stop_browser_training)
        run_actions.addWidget(self.training_stop_btn)

        self.training_build_btn = QPushButton("Build Process")
        self.training_build_btn.clicked.connect(self.build_browser_training_process)
        run_actions.addWidget(self.training_build_btn)

        self.training_run_btn = QPushButton("Run Process")
        self.training_run_btn.clicked.connect(self.run_browser_training_process)
        run_actions.addWidget(self.training_run_btn)
        layout.addLayout(run_actions)

        self.training_status = QLabel("Status: idle")
        self.training_status.setStyleSheet("color: #9fb3c8;")
        layout.addWidget(self.training_status)
        layout.addStretch(1)
        return tab

    def _build_model_api_settings_dialog(self) -> None:
        self.model_api_settings_dialog = QDialog(self)
        self.model_api_settings_dialog.setWindowTitle("Model/API Settings")
        self.model_api_settings_dialog.setMinimumWidth(860)
        self.model_api_settings_dialog.resize(980, 760)

        dialog_layout = QVBoxLayout(self.model_api_settings_dialog)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)

        ai_group = QGroupBox("AI Generation")
        ai_layout = QFormLayout(ai_group)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(os.getenv("GROK_API_KEY", ""))
        ai_layout.addRow("Grok API Key", self.api_key)

        self.chat_model = QLineEdit(os.getenv("GROK_CHAT_MODEL", "grok-3-mini"))
        ai_layout.addRow("Chat Model", self.chat_model)

        self.image_model = QLineEdit(os.getenv("GROK_VIDEO_MODEL", "grok-video-latest"))
        ai_layout.addRow("Video Model", self.image_model)

        self.prompt_source = QComboBox()
        self.prompt_source.addItem("Manual prompt (no API)", "manual")
        self.prompt_source.addItem("Grok API", "grok")
        self.prompt_source.addItem("OpenAI API", "openai")
        self.prompt_source.currentIndexChanged.connect(self._toggle_prompt_source_fields)
        ai_layout.addRow("Prompt Source", self.prompt_source)

        self.video_provider = QComboBox()
        self.video_provider.addItem("Grok Imagine API", "grok")
        self.video_provider.addItem("OpenAI Sora 2 API", "openai")
        self.video_provider.addItem("Seedance 2.0 API", "seedance")
        self.video_provider.currentIndexChanged.connect(self._toggle_prompt_source_fields)
        ai_layout.addRow("Video Provider", self.video_provider)

        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setPlaceholderText("Optional OpenAI API key (preferred when available)")
        self.openai_api_key.setText(os.getenv("OPENAI_API_KEY", ""))
        ai_layout.addRow("OpenAI API Key", self.openai_api_key)

        self.openai_access_token = QLineEdit()
        self.openai_access_token.setEchoMode(QLineEdit.Password)
        self.openai_access_token.setPlaceholderText("Optional bearer token from OAuth/browser sign-in flow")
        self.openai_access_token.setText(os.getenv("OPENAI_ACCESS_TOKEN", ""))
        ai_layout.addRow("OpenAI Access Token", self.openai_access_token)

        self.openai_chat_model = QLineEdit(os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1-codex"))
        ai_layout.addRow("OpenAI Chat Model", self.openai_chat_model)

        self.seedance_api_key = QLineEdit()
        self.seedance_api_key.setEchoMode(QLineEdit.Password)
        self.seedance_api_key.setPlaceholderText("Seedance API key")
        self.seedance_api_key.setText(os.getenv("SEEDANCE_API_KEY", ""))
        ai_layout.addRow("Seedance API Key", self.seedance_api_key)

        self.seedance_oauth_token = QLineEdit()
        self.seedance_oauth_token.setEchoMode(QLineEdit.Password)
        self.seedance_oauth_token.setPlaceholderText("Optional OAuth bearer token (if provided by Seedance)")
        self.seedance_oauth_token.setText(os.getenv("SEEDANCE_OAUTH_TOKEN", ""))
        ai_layout.addRow("Seedance OAuth Token", self.seedance_oauth_token)

        self.ai_auth_method = QComboBox()
        self.ai_auth_method.addItem("API key", "api_key")
        self.ai_auth_method.addItem("Browser sign-in (preferred)", "browser")
        ai_layout.addRow("AI Authorization", self.ai_auth_method)

        self.browser_auth_btn = QPushButton("Open Provider Login in Browser")
        self.browser_auth_btn.clicked.connect(self.open_ai_provider_login)
        ai_layout.addRow("Browser Authorization", self.browser_auth_btn)

        youtube_group = QGroupBox("YouTube")
        youtube_layout = QFormLayout(youtube_group)

        self.youtube_api_key = QLineEdit()
        self.youtube_api_key.setEchoMode(QLineEdit.Password)
        self.youtube_api_key.setText(os.getenv("YOUTUBE_API_KEY", ""))
        youtube_layout.addRow("YouTube API Key", self.youtube_api_key)

        facebook_group = QGroupBox("Facebook")
        facebook_layout = QFormLayout(facebook_group)

        self.facebook_page_id = QLineEdit(os.getenv("FACEBOOK_PAGE_ID", ""))
        facebook_layout.addRow("Facebook Page ID", self.facebook_page_id)

        self.facebook_access_token = QLineEdit()
        self.facebook_access_token.setEchoMode(QLineEdit.Password)
        self.facebook_access_token.setText(os.getenv("FACEBOOK_ACCESS_TOKEN", ""))
        facebook_layout.addRow("Facebook Access Token", self.facebook_access_token)

        self.facebook_app_id = QLineEdit(os.getenv("FACEBOOK_APP_ID", ""))
        facebook_layout.addRow("Facebook App ID", self.facebook_app_id)

        self.facebook_app_secret = QLineEdit()
        self.facebook_app_secret.setEchoMode(QLineEdit.Password)
        self.facebook_app_secret.setText(os.getenv("FACEBOOK_APP_SECRET", ""))
        facebook_layout.addRow("Facebook App Secret", self.facebook_app_secret)

        self.facebook_profile_url = QLineEdit(os.getenv("FACEBOOK_PROFILE_URL", "https://www.facebook.com/"))
        self.facebook_profile_url.setPlaceholderText("e.g. https://www.facebook.com/dave.hook.94")
        self.facebook_profile_url.setToolTip("Used for the Facebook web upload tab URL.")
        self.facebook_profile_url.editingFinished.connect(self._refresh_facebook_upload_tab_url)
        facebook_layout.addRow("Facebook Profile URL (web upload)", self.facebook_profile_url)

        self.facebook_oauth_btn = QPushButton("Authorize Facebook for Pages")
        self.facebook_oauth_btn.setToolTip("Open Facebook OAuth in browser and populate Page ID + Page access token.")
        self.facebook_oauth_btn.clicked.connect(self.authorize_facebook_pages)
        facebook_layout.addRow("Facebook OAuth", self.facebook_oauth_btn)

        instagram_group = QGroupBox("Instagram")
        instagram_layout = QFormLayout(instagram_group)

        self.instagram_business_id = QLineEdit(os.getenv("INSTAGRAM_BUSINESS_ID", ""))
        instagram_layout.addRow("Instagram Business ID", self.instagram_business_id)

        self.instagram_access_token = QLineEdit()
        self.instagram_access_token.setEchoMode(QLineEdit.Password)
        self.instagram_access_token.setText(os.getenv("INSTAGRAM_ACCESS_TOKEN", ""))
        instagram_layout.addRow("Instagram Access Token", self.instagram_access_token)

        self.instagram_username = QLineEdit(os.getenv("INSTAGRAM_USERNAME", ""))
        self.instagram_username.setPlaceholderText("e.g. funkymonk66")
        self.instagram_username.setToolTip("Used for the Instagram web upload tab URL: https://www.instagram.com/<username>/reels/create")
        self.instagram_username.editingFinished.connect(self._refresh_instagram_upload_tab_url)
        instagram_layout.addRow("Instagram Username (web upload URL)", self.instagram_username)

        tiktok_group = QGroupBox("TikTok")
        tiktok_layout = QFormLayout(tiktok_group)

        self.tiktok_access_token = QLineEdit()
        self.tiktok_access_token.setEchoMode(QLineEdit.Password)
        self.tiktok_access_token.setText(os.getenv("TIKTOK_ACCESS_TOKEN", ""))
        tiktok_layout.addRow("TikTok Access Token", self.tiktok_access_token)

        self.tiktok_client_key = QLineEdit(os.getenv("TIKTOK_CLIENT_KEY", ""))
        tiktok_layout.addRow("TikTok Client Key", self.tiktok_client_key)

        self.tiktok_client_secret = QLineEdit()
        self.tiktok_client_secret.setEchoMode(QLineEdit.Password)
        self.tiktok_client_secret.setText(os.getenv("TIKTOK_CLIENT_SECRET", ""))
        tiktok_layout.addRow("TikTok Client Secret", self.tiktok_client_secret)

        self.tiktok_oauth_btn = QPushButton("Authorize TikTok Upload")
        self.tiktok_oauth_btn.setToolTip("Open TikTok OAuth in browser and populate TikTok access token.")
        self.tiktok_oauth_btn.clicked.connect(self.authorize_tiktok_upload)
        tiktok_layout.addRow("TikTok OAuth", self.tiktok_oauth_btn)

        self.tiktok_privacy_level = QComboBox()
        self.tiktok_privacy_level.addItem("Public", "PUBLIC_TO_EVERYONE")
        self.tiktok_privacy_level.addItem("Friends (mutual follow)", "MUTUAL_FOLLOW_FRIENDS")
        self.tiktok_privacy_level.addItem("Private (only me)", "SELF_ONLY")
        env_privacy = os.getenv("TIKTOK_PRIVACY_LEVEL", "PUBLIC_TO_EVERYONE").strip().upper()
        privacy_index = self.tiktok_privacy_level.findData(env_privacy)
        if privacy_index >= 0:
            self.tiktok_privacy_level.setCurrentIndex(privacy_index)
        tiktok_layout.addRow("TikTok Privacy", self.tiktok_privacy_level)

        app_group = QGroupBox("App Preferences")
        app_layout = QFormLayout(app_group)

        self.download_path_input = QLineEdit(str(self.download_dir))
        self.download_path_input.setReadOnly(True)
        choose_download_path_btn = QPushButton("Browse...")
        choose_download_path_btn.clicked.connect(self._choose_download_path)
        download_path_row = QHBoxLayout()
        download_path_row.addWidget(self.download_path_input)
        download_path_row.addWidget(choose_download_path_btn)
        app_layout.addRow("Download Folder", download_path_row)

        self.crossfade_duration = QDoubleSpinBox()
        self.crossfade_duration.setRange(0.1, 3.0)
        self.crossfade_duration.setSingleStep(0.1)
        self.crossfade_duration.setDecimals(1)
        self.crossfade_duration.setValue(0.5)
        self.crossfade_duration.setSuffix(" s")
        self.crossfade_duration.valueChanged.connect(self._sync_video_options_label)
        app_layout.addRow("Crossfade Duration", self.crossfade_duration)

        self.manual_prompt_default_input = QPlainTextEdit()
        self.manual_prompt_default_input.setMaximumHeight(90)
        self.manual_prompt_default_input.setPlaceholderText("Default text used to prefill Manual Prompt.")
        self.manual_prompt_default_input.setPlainText(DEFAULT_MANUAL_PROMPT_TEXT)
        app_layout.addRow("Default Manual Prompt", self.manual_prompt_default_input)

        self.qtwebengine_remote_debug_enabled = QCheckBox("Enable QtWebEngine CDP remote debugging")
        self.qtwebengine_remote_debug_enabled.setChecked(_env_int("GROK_QTWEBENGINE_REMOTE_DEBUG_PORT", 0) > 0)
        app_layout.addRow("CDP Remote Debugging", self.qtwebengine_remote_debug_enabled)

        self.qtwebengine_remote_debug_port = QSpinBox()
        self.qtwebengine_remote_debug_port.setRange(1, 65535)
        self.qtwebengine_remote_debug_port.setValue(max(1, _env_int("GROK_QTWEBENGINE_REMOTE_DEBUG_PORT", 9222)))
        self.qtwebengine_remote_debug_port.setToolTip("Port used for QTWEBENGINE_REMOTE_DEBUGGING when enabled.")
        app_layout.addRow("CDP Debug Port", self.qtwebengine_remote_debug_port)

        remote_debug_note = QLabel("Applies on next app launch. Save settings, then restart the app.")
        remote_debug_note.setWordWrap(True)
        app_layout.addRow("CDP Note", remote_debug_note)

        self.cdp_social_upload_relay_enabled = QCheckBox("Use CDP relay for social browser automation")
        self.cdp_social_upload_relay_enabled.setChecked(False)
        app_layout.addRow("CDP Relay Mode", self.cdp_social_upload_relay_enabled)

        self.cdp_social_upload_relay_url = QLineEdit(CDP_RELAY_SOCIAL_UPLOAD_URL)
        self.cdp_social_upload_relay_url.setPlaceholderText("http://127.0.0.1:8765/social-upload-step")
        self.cdp_social_upload_relay_url.setToolTip("Optional HTTP relay endpoint that performs CDP-backed social upload actions.")
        self.cdp_social_upload_relay_enabled.toggled.connect(lambda _: self._reset_cdp_relay_session_state())
        self.cdp_social_upload_relay_url.editingFinished.connect(self._reset_cdp_relay_session_state)
        app_layout.addRow("CDP Relay URL", self.cdp_social_upload_relay_url)

        settings_layout.addWidget(ai_group)
        settings_layout.addWidget(youtube_group)
        settings_layout.addWidget(facebook_group)
        settings_layout.addWidget(instagram_group)
        settings_layout.addWidget(tiktok_group)
        settings_layout.addWidget(app_group)
        settings_layout.addStretch(1)

        settings_scroll.setWidget(settings_container)
        dialog_layout.addWidget(settings_scroll)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText("Save Settings")
            save_btn.clicked.connect(self.save_model_api_settings)
        close_btn = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.model_api_settings_dialog.close)
        dialog_layout.addWidget(button_box)

    def _toolbar_tinted_standard_icon(self, standard_pixmap: QStyle.StandardPixmap, color_hex: str = "#1e88e5") -> QIcon:
        base_icon = self.style().standardIcon(standard_pixmap)
        pixmap = base_icon.pixmap(20, 20)
        if pixmap.isNull():
            return base_icon

        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)

        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color_hex))
        painter.end()
        return QIcon(tinted)

    def _status_bar_has_active_content(self) -> bool:
        tracked_widgets = [
            getattr(self, "stitch_progress_label", None),
            getattr(self, "stitch_progress_bar", None),
        ]
        return any(widget is not None and widget.isVisible() for widget in tracked_widgets)

    def _start_automation_counter_tracking(self, total: int) -> None:
        self.automation_counter_total = max(0, int(total))
        self.automation_counter_completed = 0

    def _advance_automation_counter_tracking(self) -> None:
        if self.automation_counter_total <= 1:
            return
        self.automation_counter_completed = min(self.automation_counter_completed + 1, self.automation_counter_total)
        turn_text = f"{self.automation_counter_completed}/{self.automation_counter_total}"
        self._append_log(f"Automation counter progress: completed turn {turn_text}.")

    def _reset_automation_counter_tracking(self) -> None:
        self.automation_counter_total = 0
        self.automation_counter_completed = 0

    def _refresh_status_bar_visibility(self) -> None:
        status_bar = self.statusBar()
        should_show = self._status_bar_has_active_content()
        status_bar.setVisible(should_show)
        status_bar.setMaximumHeight(16777215 if should_show else 0)

    def _is_browser_tab_enabled(self, tab_key: str) -> bool:
        return bool(self.browser_tab_enabled.get(tab_key, True))

    def _set_browser_tab_enabled(self, tab_key: str, enabled: bool) -> None:
        enabled = bool(enabled)
        self.browser_tab_enabled[tab_key] = enabled

        tab_index = self.browser_tab_indices.get(tab_key) if hasattr(self, "browser_tab_indices") else None
        if tab_index is not None:
            self.browser_tabs.setTabVisible(tab_index, enabled)
            if not enabled and self.browser_tabs.currentIndex() == tab_index:
                self._refresh_browser_tab_selection()

        browser = self.browser_tab_webviews.get(tab_key) if hasattr(self, "browser_tab_webviews") else None
        if browser is not None:
            if enabled:
                browser.setEnabled(True)
            else:
                browser.stop()
                browser.setEnabled(False)

        action = self.browser_tab_toggle_actions.get(tab_key) if hasattr(self, "browser_tab_toggle_actions") else None
        if action is not None and action.isChecked() != enabled:
            action.blockSignals(True)
            action.setChecked(enabled)
            action.blockSignals(False)

    def _set_quick_actions_toolbar_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if hasattr(self, "quick_actions_toolbar"):
            self.quick_actions_toolbar.setVisible(visible)

        action = getattr(self, "quick_actions_toolbar_toggle_action", None)
        if action is not None and action.isChecked() != visible:
            action.blockSignals(True)
            action.setChecked(visible)
            action.blockSignals(False)

    def _refresh_browser_tab_selection(self) -> None:
        current = self.browser_tabs.currentIndex()
        if current >= 0 and self.browser_tabs.isTabVisible(current):
            return
        for i in range(self.browser_tabs.count()):
            if self.browser_tabs.isTabVisible(i):
                self.browser_tabs.setCurrentIndex(i)
                return

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        save_action = QAction("Save Preferences", self)
        save_action.triggered.connect(self.save_preferences)
        file_menu.addAction(save_action)

        load_action = QAction("Load Preferences...", self)
        load_action.triggered.connect(self.load_preferences)
        file_menu.addAction(load_action)

        open_video_action = QAction("Open Video(s)...", self)
        open_video_action.triggered.connect(self.open_local_video)
        file_menu.addAction(open_video_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        settings_menu = menu_bar.addMenu("Model/API Settings")
        open_settings_action = QAction("Open Model/API Settings", self)
        open_settings_action.triggered.connect(self.show_model_api_settings)
        settings_menu.addAction(open_settings_action)

        video_menu = menu_bar.addMenu("Video")
        self.video_settings_menu = video_menu.addMenu("Settings")
        self.video_grok_settings_menu = video_menu.addMenu("Grok Settings")

        audio_menu = menu_bar.addMenu("Audio")
        self.audio_settings_menu = audio_menu.addMenu("Settings")

        self.automation_menu = menu_bar.addMenu("Automation")

        cdp_settings_menu = menu_bar.addMenu("CDP Settings")
        cdp_enabled_action = QAction("Enable CDP", self)
        cdp_enabled_action.triggered.connect(lambda: self._set_cdp_enabled(True))
        cdp_settings_menu.addAction(cdp_enabled_action)

        cdp_disabled_action = QAction("Disable CDP", self)
        cdp_disabled_action.triggered.connect(lambda: self._set_cdp_enabled(False))
        cdp_settings_menu.addAction(cdp_disabled_action)

        self.cdp_menu_actions = {
            True: cdp_enabled_action,
            False: cdp_disabled_action,
        }
        self._set_cdp_enabled(self.cdp_enabled)

        view_menu = menu_bar.addMenu("View")
        self.quick_actions_toolbar_toggle_action = QAction("Show Top Icon Toolbar", self)
        self.quick_actions_toolbar_toggle_action.setCheckable(True)
        self.quick_actions_toolbar_toggle_action.setChecked(True)
        self.quick_actions_toolbar_toggle_action.toggled.connect(self._set_quick_actions_toolbar_visible)
        view_menu.addAction(self.quick_actions_toolbar_toggle_action)

        browser_tabs_menu = view_menu.addMenu("Browser Tabs")
        self.browser_tab_toggle_actions = {}
        browser_tab_menu_items = (
            ("Grok", "Grok"),
            ("Sora", "Sora"),
            ("Facebook", "Facebook"),
            ("Instagram", "Instagram"),
            ("TikTok", "TikTok"),
            ("YouTube", "YouTube"),
            ("Sora2Settings", "Sora 2 Video Settings"),
            ("SeedanceSettings", "Seedance 2.0 Video Settings"),
            ("AIFlowTrainer", "AI Flow Trainer"),
        )
        for tab_key, tab_label in browser_tab_menu_items:
            action = QAction(tab_label, self)
            action.setCheckable(True)
            action.setChecked(self._is_browser_tab_enabled(tab_key))
            action.toggled.connect(lambda checked, k=tab_key: self._set_browser_tab_enabled(k, checked))
            browser_tabs_menu.addAction(action)
            self.browser_tab_toggle_actions[tab_key] = action

        help_menu = menu_bar.addMenu("Help")
        info_action = QAction("Info", self)
        info_action.triggered.connect(self.show_app_info)
        help_menu.addAction(info_action)

        github_action = QAction("GitHub", self)
        github_action.triggered.connect(self.open_github_page)
        help_menu.addAction(github_action)

        releases_action = QAction("Download Windows Binary", self)
        releases_action.triggered.connect(self.open_github_releases_page)
        help_menu.addAction(releases_action)

        check_updates_action = QAction("Check for Updates", self)
        check_updates_action.triggered.connect(lambda: self.check_for_updates(manual=True))
        help_menu.addAction(check_updates_action)

        actions_action = QAction("Build Artifacts", self)
        actions_action.triggered.connect(self.open_github_actions_runs_page)
        help_menu.addAction(actions_action)

        extension_dir_action = QAction("Open Chrome Extension Directory", self)
        extension_dir_action.triggered.connect(self.open_extension_directory)
        help_menu.addAction(extension_dir_action)

        downloads_dir_action = QAction("Open Downloads Folder", self)
        downloads_dir_action.triggered.connect(self.open_downloads_folder)
        help_menu.addAction(downloads_dir_action)

        self.quick_actions_toolbar = QToolBar("Quick Actions", self)
        self.quick_actions_toolbar.setMovable(False)
        self.quick_actions_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.quick_actions_toolbar)

        stop_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserStop), "Stop All Jobs", self)
        stop_action.setToolTip("Stop all active generation jobs.")
        stop_action.triggered.connect(self.stop_all_jobs)
        self.quick_actions_toolbar.addAction(stop_action)

        create_video_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Create New Video", self)
        create_video_action.setToolTip("Create a new video from the current prompt in Grok.")
        create_video_action.triggered.connect(self.start_image_generation)
        self.quick_actions_toolbar.addAction(create_video_action)

        continue_last_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward), "Continue Last Video", self)
        continue_last_action.setToolTip("Continue from the last generated video's final frame.")
        continue_last_action.triggered.connect(self.continue_from_last_frame)
        self.quick_actions_toolbar.addAction(continue_last_action)

        continue_image_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Create From Image", self)
        continue_image_action.setToolTip("Choose a local image and generate a new continuation video.")
        continue_image_action.triggered.connect(self.continue_from_local_image)
        self.quick_actions_toolbar.addAction(continue_image_action)

        self.quick_actions_toolbar.addSeparator()
        homepage_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon), "Open Grok Homepage", self)
        homepage_action.setToolTip("Open grok.com/imagine in the embedded browser.")
        homepage_action.triggered.connect(self.show_browser_page)
        self.quick_actions_toolbar.addAction(homepage_action)


        open_downloads_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), "Open Downloads Folder", self)
        open_downloads_action.setToolTip("Open the downloads folder in your system file explorer.")
        open_downloads_action.triggered.connect(self.open_downloads_folder)
        self.quick_actions_toolbar.addAction(open_downloads_action)

        self.quick_actions_toolbar.addSeparator()
        auto_youtube_upload_action = QAction(self._toolbar_tinted_standard_icon(QStyle.StandardPixmap.SP_ArrowUp), "Automate YouTube Upload", self)
        auto_youtube_upload_action.setToolTip("Run browser automation for YouTube upload in the YouTube tab.")
        auto_youtube_upload_action.triggered.connect(self.start_youtube_browser_upload)
        self.quick_actions_toolbar.addAction(auto_youtube_upload_action)

        auto_facebook_upload_action = QAction(self._toolbar_tinted_standard_icon(QStyle.StandardPixmap.SP_ArrowUp), "Automate Facebook Upload", self)
        auto_facebook_upload_action.setToolTip("Run browser automation for Facebook upload in the Facebook tab.")
        auto_facebook_upload_action.triggered.connect(self.start_facebook_browser_upload)
        self.quick_actions_toolbar.addAction(auto_facebook_upload_action)

        auto_instagram_upload_action = QAction(self._toolbar_tinted_standard_icon(QStyle.StandardPixmap.SP_ArrowUp), "Automate Instagram Upload", self)
        auto_instagram_upload_action.setToolTip("Run browser automation for Instagram upload in the Instagram tab.")
        auto_instagram_upload_action.triggered.connect(self.start_instagram_browser_upload)
        self.quick_actions_toolbar.addAction(auto_instagram_upload_action)

        auto_tiktok_upload_action = QAction(self._toolbar_tinted_standard_icon(QStyle.StandardPixmap.SP_ArrowUp), "Automate TikTok Upload", self)
        auto_tiktok_upload_action.setToolTip("Run browser automation for TikTok upload in the TikTok tab.")
        auto_tiktok_upload_action.triggered.connect(self.start_tiktok_browser_upload)
        self.quick_actions_toolbar.addAction(auto_tiktok_upload_action)

        self.quick_actions_toolbar.addSeparator()
        youtube_upload_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Upload to YouTube", self)
        youtube_upload_action.setToolTip("Upload selected video to YouTube.")
        youtube_upload_action.triggered.connect(self.upload_selected_to_youtube)
        self.quick_actions_toolbar.addAction(youtube_upload_action)

        facebook_upload_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Upload to Facebook", self)
        facebook_upload_action.setToolTip("Upload selected video to Facebook.")
        facebook_upload_action.triggered.connect(self.upload_selected_to_facebook)
        self.quick_actions_toolbar.addAction(facebook_upload_action)

        instagram_upload_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Upload to Instagram", self)
        instagram_upload_action.setToolTip("Upload selected video to Instagram.")
        instagram_upload_action.triggered.connect(self.upload_selected_to_instagram)
        self.quick_actions_toolbar.addAction(instagram_upload_action)

        tiktok_upload_action = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), "Upload to TikTok", self)
        tiktok_upload_action.setToolTip("Upload selected video to TikTok.")
        tiktok_upload_action.triggered.connect(self.upload_selected_to_tiktok)
        self.quick_actions_toolbar.addAction(tiktok_upload_action)

        self._set_quick_actions_toolbar_visible(self.quick_actions_toolbar_toggle_action.isChecked())

    def _run_with_button_feedback(self, button: QPushButton, callback: Callable[[], None]) -> None:
        if button is not None and button.isCheckable():
            button.setChecked(True)
            QTimer.singleShot(220, lambda: button.setChecked(False))
        callback()

    def _add_widget_to_menu(self, menu: QMenu, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(widget)
        menu.addAction(action)

    def _populate_top_settings_menus(self) -> None:
        self.video_settings_menu.clear()
        self.video_grok_settings_menu.clear()
        self.audio_settings_menu.clear()
        self.automation_menu.clear()

        self._add_widget_to_menu(self.video_settings_menu, self.stitch_crossfade_checkbox)
        self._add_widget_to_menu(self.video_settings_menu, self.video_options_dropdown)
        self.video_settings_menu.addSeparator()
        self._add_widget_to_menu(self.video_settings_menu, self.stitch_interpolation_checkbox)
        self._add_widget_to_menu(self.video_settings_menu, self.stitch_interpolation_fps)
        self.video_settings_menu.addSeparator()
        self._add_widget_to_menu(self.video_settings_menu, self.stitch_upscale_checkbox)
        self._add_widget_to_menu(self.video_settings_menu, self.stitch_upscale_target)
        self.video_settings_menu.addSeparator()
        self._add_widget_to_menu(self.video_settings_menu, self.stitch_gpu_checkbox)

        grok_resolution_widget = QWidget(self)
        grok_resolution_layout = QHBoxLayout(grok_resolution_widget)
        grok_resolution_layout.setContentsMargins(8, 4, 8, 4)
        grok_resolution_layout.addWidget(QLabel("Resolution"))
        grok_resolution_layout.addWidget(self.video_resolution)
        self._add_widget_to_menu(self.video_grok_settings_menu, grok_resolution_widget)

        grok_duration_widget = QWidget(self)
        grok_duration_layout = QHBoxLayout(grok_duration_widget)
        grok_duration_layout.setContentsMargins(8, 4, 8, 4)
        grok_duration_layout.addWidget(QLabel("Duration"))
        grok_duration_layout.addWidget(self.video_duration)
        self._add_widget_to_menu(self.video_grok_settings_menu, grok_duration_widget)

        grok_aspect_widget = QWidget(self)
        grok_aspect_layout = QHBoxLayout(grok_aspect_widget)
        grok_aspect_layout.setContentsMargins(8, 4, 8, 4)
        grok_aspect_layout.addWidget(QLabel("Aspect"))
        grok_aspect_layout.addWidget(self.video_aspect_ratio)
        self._add_widget_to_menu(self.video_grok_settings_menu, grok_aspect_widget)

        self._add_widget_to_menu(self.audio_settings_menu, self.stitch_mute_original_checkbox)
        self.audio_settings_menu.addSeparator()
        self._add_widget_to_menu(self.audio_settings_menu, self.stitch_original_audio_volume)
        self._add_widget_to_menu(self.audio_settings_menu, self.stitch_music_volume)
        self._add_widget_to_menu(self.audio_settings_menu, self.stitch_audio_fade_duration)
        self._add_widget_to_menu(self.audio_settings_menu, self.stitch_audio_fade_label)

        automation_widget = QWidget(self)
        automation_layout = QHBoxLayout(automation_widget)
        automation_layout.setContentsMargins(8, 4, 8, 4)
        automation_layout.addWidget(QLabel("Counter"))
        automation_layout.addWidget(self.count)
        automation_layout.addStretch(1)
        self._add_widget_to_menu(self.automation_menu, automation_widget)

    def show_app_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("App Info")
        dialog.setMinimumWidth(680)

        info_browser = QTextBrowser(dialog)
        info_browser.setOpenExternalLinks(True)
        info_browser.setReadOnly(True)
        info_browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        info_browser.setHtml(
            "<h3>Grok Video Desktop Studio</h3>"
            f"<p><b>Version:</b> {CURRENT_APP_VERSION}<br>"
            "<b>Authors:</b> Grok Video Desktop Studio contributors<br>"
            "<b>Desktop workflow:</b> PyQt + embedded Grok browser + YouTube uploader</p>"
            "<p><b>Downloads</b><br>"
            f"- Releases: <a href='{GITHUB_RELEASES_URL}'>{GITHUB_RELEASES_URL}</a><br>"
            f"- CI workflow artifacts: <a href='{GITHUB_ACTIONS_RUNS_URL}'>{GITHUB_ACTIONS_RUNS_URL}</a></p>"
            "<p>If this saves you hours, grab me a ☕.</p>"
            "<p><b>Support links</b><br>"
            f"- Buy Me a Coffee: <a href='{BUY_ME_A_COFFEE_URL}'>{BUY_ME_A_COFFEE_URL}</a><br>"
            f"- PayPal: <a href='{PAYPAL_DONATION_URL}'>{PAYPAL_DONATION_URL}</a><br>"
            f"- Crypto (SOL): <code>{SOL_DONATION_ADDRESS}</code></p>"
        )

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        button_box.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dialog.close)

        layout = QVBoxLayout(dialog)
        layout.addWidget(info_browser)
        layout.addWidget(button_box)

        dialog.exec()

    def open_github_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL))

    def open_github_releases_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_RELEASES_URL))

    def open_github_actions_runs_page(self) -> None:
        QDesktopServices.openUrl(QUrl(GITHUB_ACTIONS_RUNS_URL))

    def _release_asset_matches_platform(self, asset_name: str) -> bool:
        normalized = str(asset_name or "").strip().lower()
        if not normalized:
            return False
        if sys.platform.startswith("win"):
            return any(token in normalized for token in (".exe", ".msi", "windows", "win64", "win32"))
        if sys.platform == "darwin":
            return any(token in normalized for token in (".dmg", ".pkg", "mac", "darwin", "osx"))
        return any(token in normalized for token in (".appimage", ".deb", ".rpm", ".tar.gz", ".zip", "linux"))

    def _fetch_latest_release(self) -> dict[str, object] | None:
        try:
            response = requests.get(
                GITHUB_LATEST_RELEASE_API_URL,
                headers={"Accept": "application/vnd.github+json"},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            self._append_log(f"Update check failed: {exc}")
        return None

    def _pick_installer_asset(self, release_payload: dict[str, object]) -> dict[str, str] | None:
        assets = release_payload.get("assets")
        if not isinstance(assets, list):
            return None
        candidates: list[dict[str, str]] = []
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name") or "").strip()
            url = str(asset.get("browser_download_url") or "").strip()
            if not name or not url:
                continue
            if self._release_asset_matches_platform(name):
                candidates.append({"name": name, "url": url})
        if not candidates:
            return None
        candidates.sort(key=lambda item: len(item["name"]))
        return candidates[0]

    def _download_update_installer(self, download_url: str, filename_hint: str) -> Path:
        updates_dir = self.download_dir / "updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        target_path = updates_dir / _slugify_filename_part(Path(filename_hint).stem, "installer")
        suffix = Path(filename_hint).suffix or ".bin"
        final_path = target_path.with_suffix(suffix)

        with requests.get(download_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(final_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        handle.write(chunk)
        return final_path

    def _open_installer_file(self, installer_path: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(installer_path))  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(installer_path)])
            return
        subprocess.Popen(["xdg-open", str(installer_path)])

    def _maybe_run_update_installation(self, release_payload: dict[str, object]) -> None:
        installer_asset = self._pick_installer_asset(release_payload)
        if installer_asset is None:
            QMessageBox.information(
                self,
                "Installer Not Found",
                "No installer was found for your operating system in the latest release assets.",
            )
            self.open_github_releases_page()
            return

        installer_name = installer_asset["name"]
        installer_url = installer_asset["url"]
        self._append_log(f"Downloading update installer: {installer_name}")
        try:
            installer_path = self._download_update_installer(installer_url, installer_name)
            self._append_log(f"Downloaded update installer to: {installer_path}")
            self._open_installer_file(installer_path)
        except Exception as exc:
            QMessageBox.warning(self, "Update Download Failed", f"Could not download installer.\n\n{exc}")
            self._append_log(f"Update installer download/open failed: {exc}")
            return

        QMessageBox.information(
            self,
            "Installing Update",
            "The installer has been opened. The app will now close so the update can continue.",
        )
        self.close()

    def check_for_updates_on_startup(self) -> None:
        self.check_for_updates(manual=False)

    def check_for_updates(self, manual: bool = False) -> None:
        now_ts = int(time.time())
        one_day_seconds = 24 * 60 * 60
        if not manual:
            last_prompt = int(getattr(self, "last_update_prompt_ts", 0) or 0)
            if last_prompt > 0 and (now_ts - last_prompt) < one_day_seconds:
                return

        release_payload = self._fetch_latest_release()
        if release_payload is None:
            if manual:
                QMessageBox.warning(self, "Update Check Failed", "Could not reach GitHub releases right now.")
            return

        latest_version = str(release_payload.get("tag_name") or "").strip() or str(release_payload.get("name") or "").strip()
        if not latest_version:
            if manual:
                QMessageBox.warning(self, "Update Check Failed", "Latest release version could not be determined.")
            return

        current_key = _normalize_version_key(CURRENT_APP_VERSION)
        latest_key = _normalize_version_key(latest_version)
        if latest_key <= current_key:
            if manual:
                QMessageBox.information(
                    self,
                    "No Update Needed",
                    f"You are running version {CURRENT_APP_VERSION}, which is up to date.",
                )
            return

        self.last_update_prompt_ts = now_ts
        self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=False, only_if_changed=False)

        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Icon.Information)
        prompt.setWindowTitle("Update Available")
        prompt.setText(
            f"A newer version is available.\n\nCurrent: {CURRENT_APP_VERSION}\nLatest: {latest_version}\n\n"
            "Would you like to download and install the latest recommended version now?"
        )
        prompt.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        prompt.setDefaultButton(QMessageBox.StandardButton.Yes)
        if prompt.exec() == QMessageBox.StandardButton.Yes:
            self._maybe_run_update_installation(release_payload)

    def open_extension_directory(self) -> None:
        extension_dir = (BASE_DIR / "extension").resolve()
        QMessageBox.information(
            self,
            "Chrome Extension Directory",
            f"Chrome extension path:\n{extension_dir}",
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(extension_dir)))

    def open_downloads_folder(self) -> None:
        download_path = str(self.download_dir.resolve())
        if sys.platform.startswith("win"):
            try:
                subprocess.Popen(["explorer", download_path])
            except Exception:
                QDesktopServices.openUrl(QUrl.fromLocalFile(download_path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(download_path))

    def _set_cdp_enabled(self, enabled: bool) -> None:
        self.cdp_enabled = bool(enabled)
        if hasattr(self, "automation_group"):
            self.automation_group.setVisible(self.cdp_enabled)
        if hasattr(self, "cdp_menu_actions"):
            self.cdp_menu_actions[True].setEnabled(not self.cdp_enabled)
            self.cdp_menu_actions[False].setEnabled(self.cdp_enabled)

    def open_buy_me_a_coffee(self) -> None:
        QDesktopServices.openUrl(QUrl(BUY_ME_A_COFFEE_URL))

    def show_model_api_settings(self) -> None:
        self.model_api_settings_dialog.show()
        self.model_api_settings_dialog.raise_()
        self.model_api_settings_dialog.activateWindow()

    def _choose_custom_music_file(self) -> None:
        music_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background music",
            str(self.download_dir),
            "Audio Files (*.mp3 *.wav)",
        )
        if not music_path:
            return

        selected_path = Path(music_path)
        if not selected_path.exists():
            QMessageBox.warning(self, "Music Missing", "Selected music file was not found on disk.")
            return

        self.custom_music_file = selected_path
        self.music_file_label.setText(f"Music: {selected_path.name}")
        self._append_log(f"Selected custom stitch music: {selected_path}")

    def _clear_custom_music_file(self) -> None:
        self.custom_music_file = None
        self.music_file_label.setText("Music: none selected")
        self._append_log("Cleared custom stitch music selection.")

    def _choose_download_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose Download Folder", str(self.download_dir))
        if not path:
            return
        self.download_dir = Path(path)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.download_path_input.setText(str(self.download_dir))
        self._append_log(f"Download folder set to: {self.download_dir}")

    def _on_video_options_selected(self, index: int) -> None:
        option_value = self.video_options_dropdown.itemData(index)
        if option_value is None:
            self.show_model_api_settings()
            self._sync_video_options_label()
            return

        try:
            self.crossfade_duration.setValue(float(option_value))
            self._sync_video_options_label()
        except (TypeError, ValueError):
            pass

    def _sync_video_options_label(self) -> None:
        duration = self.crossfade_duration.value()
        target_index = self.video_options_dropdown.findData(duration)
        self.video_options_dropdown.blockSignals(True)
        if target_index >= 0:
            self.video_options_dropdown.setCurrentIndex(target_index)
        else:
            self.video_options_dropdown.setCurrentIndex(2)
        self.video_options_dropdown.blockSignals(False)

    def _initialize_preferences_autosave(self) -> None:
        self._preferences_autosave_timer = QTimer(self)
        self._preferences_autosave_timer.setSingleShot(True)
        self._preferences_autosave_timer.setInterval(700)
        self._preferences_autosave_timer.timeout.connect(self._autosave_preferences_to_default_file)

        connected_widgets: set[int] = set()

        def _connect(widget: QWidget, signal_name: str) -> None:
            key = (id(widget), signal_name)
            if key in connected_widgets:
                return
            signal = getattr(widget, signal_name, None)
            if signal is None:
                return
            signal.connect(self._schedule_preferences_autosave)
            connected_widgets.add(key)

        for widget in self.findChildren(QLineEdit):
            _connect(widget, "editingFinished")
        for widget in self.findChildren(QPlainTextEdit):
            _connect(widget, "textChanged")
        for widget in self.findChildren(QTextEdit):
            _connect(widget, "textChanged")
        for widget in self.findChildren(QComboBox):
            _connect(widget, "currentIndexChanged")
        for widget in self.findChildren(QCheckBox):
            _connect(widget, "toggled")
        for widget in self.findChildren(QSpinBox):
            _connect(widget, "valueChanged")
        for widget in self.findChildren(QDoubleSpinBox):
            _connect(widget, "valueChanged")
        for widget in self.findChildren(QSlider):
            _connect(widget, "valueChanged")
        for widget in self.findChildren(QFontComboBox):
            _connect(widget, "currentFontChanged")

    def _schedule_preferences_autosave(self, *_args: object) -> None:
        if self._applying_preferences:
            return
        self._preferences_autosave_timer.start()

    def _autosave_preferences_to_default_file(self) -> None:
        self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=False, only_if_changed=True)

    @staticmethod
    def _preferences_signature(preferences: dict) -> str:
        return json.dumps(preferences, sort_keys=True)

    def _collect_preferences(self) -> dict:
        return {
            "api_key": self.api_key.text(),
            "chat_model": self.chat_model.text(),
            "image_model": self.image_model.text(),
            "prompt_source": self.prompt_source.currentData(),
            "video_provider": self.video_provider.currentData(),
            "openai_api_key": self.openai_api_key.text(),
            "openai_access_token": self.openai_access_token.text(),
            "openai_chat_model": self.openai_chat_model.text(),
            "seedance_api_key": self.seedance_api_key.text(),
            "seedance_oauth_token": self.seedance_oauth_token.text(),
            "ai_auth_method": self.ai_auth_method.currentData(),
            "youtube_api_key": self.youtube_api_key.text(),
            "facebook_page_id": self.facebook_page_id.text(),
            "facebook_access_token": self.facebook_access_token.text(),
            "facebook_app_id": self.facebook_app_id.text(),
            "facebook_app_secret": self.facebook_app_secret.text(),
            "facebook_profile_url": self.facebook_profile_url.text(),
            "instagram_business_id": self.instagram_business_id.text(),
            "instagram_access_token": self.instagram_access_token.text(),
            "instagram_username": self.instagram_username.text(),
            "tiktok_access_token": self.tiktok_access_token.text(),
            "tiktok_client_key": self.tiktok_client_key.text(),
            "tiktok_client_secret": self.tiktok_client_secret.text(),
            "tiktok_privacy_level": self.tiktok_privacy_level.currentData(),
            "concept": self.concept.toPlainText(),
            "manual_prompt": self.manual_prompt.toPlainText(),
            "manual_prompt_default": self.manual_prompt_default_input.toPlainText(),
            "qtwebengine_remote_debug_enabled": self.qtwebengine_remote_debug_enabled.isChecked(),
            "qtwebengine_remote_debug_port": int(self.qtwebengine_remote_debug_port.value()),
            "cdp_social_upload_relay_enabled": self.cdp_social_upload_relay_enabled.isChecked(),
            "cdp_social_upload_relay_url": self.cdp_social_upload_relay_url.text().strip(),
            "cdp_enabled": self.cdp_enabled,
            "browser_tab_enabled": dict(self.browser_tab_enabled),
            "quick_actions_toolbar_visible": self.quick_actions_toolbar.isVisible(),
            "automation_mode": self.automation_mode.currentData(),
            "counter": self.count.value(),
            "video_resolution": str(self.video_resolution.currentData()),
            "video_duration_seconds": int(self.video_duration.currentData()),
            "video_aspect_ratio": str(self.video_aspect_ratio.currentData()),
            "sora2_model": self.sora2_model.currentData() or self.sora2_model.currentText(),
            "sora2_seconds": str(self.sora2_seconds.currentData()),
            "sora2_size": str(self.sora2_size.currentData()),
            "sora2_input_reference": self.sora2_input_reference.text(),
            "sora2_continue_from_last_frame": self.sora2_continue_from_last_frame.isChecked(),
            "sora2_extra_body": self.sora2_extra_body.toPlainText(),
            "seedance_model": self.seedance_model.currentData() or self.seedance_model.currentText(),
            "seedance_resolution": str(self.seedance_resolution.currentData()),
            "seedance_aspect_ratio": str(self.seedance_aspect_ratio.currentData()),
            "seedance_duration_seconds": str(self.seedance_duration_seconds.currentData()),
            "seedance_fps": str(self.seedance_fps.currentData()),
            "seedance_motion_strength": self.seedance_motion_strength.value(),
            "seedance_guidance_scale": self.seedance_guidance_scale.value(),
            "seedance_seed": self.seedance_seed.text(),
            "seedance_negative_prompt": self.seedance_negative_prompt.text(),
            "seedance_camera_motion": str(self.seedance_camera_motion.currentData()),
            "seedance_style_preset": str(self.seedance_style_preset.currentData()),
            "seedance_watermark": self.seedance_watermark.isChecked(),
            "seedance_extra_body": self.seedance_extra_body.toPlainText(),
            "stitch_crossfade_enabled": self.stitch_crossfade_checkbox.isChecked(),
            "stitch_interpolation_enabled": self.stitch_interpolation_checkbox.isChecked(),
            "stitch_interpolation_fps": int(self.stitch_interpolation_fps.currentData()),
            "stitch_upscale_enabled": self.stitch_upscale_checkbox.isChecked(),
            "stitch_upscale_target": str(self.stitch_upscale_target.currentData()),
            "stitch_gpu_enabled": self.stitch_gpu_checkbox.isChecked(),
            "stitch_mute_original_audio": self.stitch_mute_original_checkbox.isChecked(),
            "stitch_original_audio_volume": self.stitch_original_audio_volume.value(),
            "stitch_music_volume": self.stitch_music_volume.value(),
            "stitch_audio_fade_duration": self.stitch_audio_fade_duration.value(),
            "stitch_custom_music_file": str(self.custom_music_file) if self.custom_music_file else "",
            "crossfade_duration": self.crossfade_duration.value(),
            "download_dir": str(self.download_dir),
            "preview_muted": self.preview_mute_checkbox.isChecked(),
            "preview_volume": self.preview_volume_slider.value(),
            "training_start_url": self.training_start_url.text(),
            "training_output_dir": self.training_output_dir.text(),
            "training_timeout": self.training_timeout.value(),
            "training_openai_model": self.training_openai_model.text(),
            "training_use_embedded_browser": self.training_use_embedded_browser.isChecked(),
            "training_trace_path": self.training_trace_path.text(),
            "training_process_path": self.training_process_path.text(),
            "last_update_prompt_ts": int(getattr(self, "last_update_prompt_ts", 0) or 0),
            "generated_videos": self._serialize_video_list_for_preferences(),
            "selected_generated_video_index": self._selected_video_index(),
            "ai_social_metadata": {
                "title": self.ai_social_metadata.title,
                "medium_title": self.ai_social_metadata.medium_title,
                "tiktok_subheading": self.ai_social_metadata.tiktok_subheading,
                "description": self.ai_social_metadata.description,
                "hashtags": self.ai_social_metadata.hashtags,
                "category": self.ai_social_metadata.category,
            },
        }

    def _apply_preferences(self, preferences: dict) -> None:
        if not isinstance(preferences, dict):
            raise ValueError("Preferences file must contain a JSON object.")

        self._applying_preferences = True
        if "api_key" in preferences:
            self.api_key.setText(str(preferences["api_key"]))
        if "chat_model" in preferences:
            self.chat_model.setText(str(preferences["chat_model"]))
        if "image_model" in preferences:
            self.image_model.setText(str(preferences["image_model"]))
        if "prompt_source" in preferences:
            source_index = self.prompt_source.findData(str(preferences["prompt_source"]))
            if source_index >= 0:
                self.prompt_source.setCurrentIndex(source_index)
        if "video_provider" in preferences:
            provider_index = self.video_provider.findData(str(preferences["video_provider"]))
            if provider_index >= 0:
                self.video_provider.setCurrentIndex(provider_index)
        if "openai_api_key" in preferences:
            self.openai_api_key.setText(str(preferences["openai_api_key"]))
        if "openai_access_token" in preferences:
            self.openai_access_token.setText(str(preferences["openai_access_token"]))
        if "openai_chat_model" in preferences:
            self.openai_chat_model.setText(str(preferences["openai_chat_model"]))
        if "seedance_api_key" in preferences:
            self.seedance_api_key.setText(str(preferences["seedance_api_key"]))
        if "seedance_oauth_token" in preferences:
            self.seedance_oauth_token.setText(str(preferences["seedance_oauth_token"]))
        if "ai_auth_method" in preferences:
            auth_index = self.ai_auth_method.findData(str(preferences["ai_auth_method"]))
            if auth_index >= 0:
                self.ai_auth_method.setCurrentIndex(auth_index)
        if "youtube_api_key" in preferences:
            self.youtube_api_key.setText(str(preferences["youtube_api_key"]))
        if "facebook_page_id" in preferences:
            self.facebook_page_id.setText(str(preferences["facebook_page_id"]))
        if "facebook_access_token" in preferences:
            self.facebook_access_token.setText(str(preferences["facebook_access_token"]))
        if "facebook_app_id" in preferences:
            self.facebook_app_id.setText(str(preferences["facebook_app_id"]))
        if "facebook_app_secret" in preferences:
            self.facebook_app_secret.setText(str(preferences["facebook_app_secret"]))
        if "facebook_profile_url" in preferences:
            self.facebook_profile_url.setText(str(preferences["facebook_profile_url"]))
            self._refresh_facebook_upload_tab_url()
        if "instagram_business_id" in preferences:
            self.instagram_business_id.setText(str(preferences["instagram_business_id"]))
        if "instagram_access_token" in preferences:
            self.instagram_access_token.setText(str(preferences["instagram_access_token"]))
        if "instagram_username" in preferences:
            self.instagram_username.setText(str(preferences["instagram_username"]))
            self._refresh_instagram_upload_tab_url()
        if "tiktok_access_token" in preferences:
            self.tiktok_access_token.setText(str(preferences["tiktok_access_token"]))
        if "tiktok_client_key" in preferences:
            self.tiktok_client_key.setText(str(preferences["tiktok_client_key"]))
        if "tiktok_client_secret" in preferences:
            self.tiktok_client_secret.setText(str(preferences["tiktok_client_secret"]))
        if "tiktok_privacy_level" in preferences:
            privacy_index = self.tiktok_privacy_level.findData(str(preferences["tiktok_privacy_level"]))
            if privacy_index >= 0:
                self.tiktok_privacy_level.setCurrentIndex(privacy_index)
        if "concept" in preferences:
            self.concept.setPlainText(str(preferences["concept"]))
        if "manual_prompt" in preferences:
            self.manual_prompt.setPlainText(str(preferences["manual_prompt"]))
        if "manual_prompt_default" in preferences:
            default_prompt = str(preferences["manual_prompt_default"])
            self.manual_prompt_default_input.setPlainText(default_prompt)
            if "manual_prompt" not in preferences:
                self.manual_prompt.setPlainText(default_prompt)
        if "qtwebengine_remote_debug_enabled" in preferences:
            self.qtwebengine_remote_debug_enabled.setChecked(bool(preferences["qtwebengine_remote_debug_enabled"]))
        if "qtwebengine_remote_debug_port" in preferences:
            try:
                port_value = int(preferences["qtwebengine_remote_debug_port"])
                if port_value > 0:
                    self.qtwebengine_remote_debug_port.setValue(port_value)
            except (TypeError, ValueError):
                pass
        if "cdp_social_upload_relay_enabled" in preferences:
            self.cdp_social_upload_relay_enabled.setChecked(bool(preferences["cdp_social_upload_relay_enabled"]))
        if "cdp_social_upload_relay_url" in preferences:
            self.cdp_social_upload_relay_url.setText(str(preferences["cdp_social_upload_relay_url"]))
        if "cdp_enabled" in preferences:
            self._set_cdp_enabled(bool(preferences["cdp_enabled"]))
        if "browser_tab_enabled" in preferences and isinstance(preferences["browser_tab_enabled"], dict):
            for tab_key, enabled in preferences["browser_tab_enabled"].items():
                if tab_key in self.browser_tab_enabled:
                    self.browser_tab_enabled[tab_key] = bool(enabled)
            if hasattr(self, "browser_tab_indices"):
                for tab_key in self.browser_tab_indices:
                    self._set_browser_tab_enabled(tab_key, self._is_browser_tab_enabled(tab_key))
                self._refresh_browser_tab_selection()
        if "quick_actions_toolbar_visible" in preferences:
            self._set_quick_actions_toolbar_visible(bool(preferences["quick_actions_toolbar_visible"]))
        if "automation_mode" in preferences:
            automation_mode_index = self.automation_mode.findData(str(preferences["automation_mode"]))
            if automation_mode_index >= 0:
                self.automation_mode.setCurrentIndex(automation_mode_index)
        if "ai_social_metadata" in preferences and isinstance(preferences["ai_social_metadata"], dict):
            metadata = preferences["ai_social_metadata"]
            hashtags = metadata.get("hashtags", self.ai_social_metadata.hashtags)
            self.ai_social_metadata = AISocialMetadata(
                title=str(metadata.get("title", self.ai_social_metadata.title)),
                medium_title=str(metadata.get("medium_title", self.ai_social_metadata.medium_title)),
                tiktok_subheading=str(metadata.get("tiktok_subheading", self.ai_social_metadata.tiktok_subheading)),
                description=str(metadata.get("description", self.ai_social_metadata.description)),
                hashtags=[str(tag).strip().lstrip("#") for tag in hashtags if str(tag).strip()],
                category=str(metadata.get("category", self.ai_social_metadata.category)),
            )
        if "counter" in preferences:
            try:
                self.count.setValue(int(preferences["counter"]))
            except (TypeError, ValueError):
                pass
        elif "count" in preferences:
            try:
                self.count.setValue(int(preferences["count"]))
            except (TypeError, ValueError):
                pass
        if "video_resolution" in preferences:
            resolution_index = self.video_resolution.findData(str(preferences["video_resolution"]))
            if resolution_index >= 0:
                self.video_resolution.setCurrentIndex(resolution_index)
        if "video_duration_seconds" in preferences:
            try:
                duration_value = int(preferences["video_duration_seconds"])
                duration_index = self.video_duration.findData(duration_value)
                if duration_index >= 0:
                    self.video_duration.setCurrentIndex(duration_index)
            except (TypeError, ValueError):
                pass
        if "video_aspect_ratio" in preferences:
            aspect_index = self.video_aspect_ratio.findData(str(preferences["video_aspect_ratio"]))
            if aspect_index >= 0:
                self.video_aspect_ratio.setCurrentIndex(aspect_index)

        if "sora2_model" in preferences:
            sora_model = str(preferences["sora2_model"])
            model_index = self.sora2_model.findData(sora_model)
            if model_index >= 0:
                self.sora2_model.setCurrentIndex(model_index)
            else:
                self.sora2_model.setCurrentText(sora_model)
        if "sora2_seconds" in preferences:
            seconds_index = self.sora2_seconds.findData(str(preferences["sora2_seconds"]))
            if seconds_index >= 0:
                self.sora2_seconds.setCurrentIndex(seconds_index)
        if "sora2_size" in preferences:
            size_index = self.sora2_size.findData(str(preferences["sora2_size"]))
            if size_index >= 0:
                self.sora2_size.setCurrentIndex(size_index)
        if "sora2_input_reference" in preferences:
            self.sora2_input_reference.setText(str(preferences["sora2_input_reference"]))
        if "sora2_continue_from_last_frame" in preferences:
            self.sora2_continue_from_last_frame.setChecked(bool(preferences["sora2_continue_from_last_frame"]))
        if "sora2_extra_body" in preferences:
            self.sora2_extra_body.setPlainText(str(preferences["sora2_extra_body"]))
        if "seedance_model" in preferences:
            idx = self.seedance_model.findData(str(preferences["seedance_model"]))
            if idx >= 0:
                self.seedance_model.setCurrentIndex(idx)
            else:
                self.seedance_model.setCurrentText(str(preferences["seedance_model"]))
        if "seedance_resolution" in preferences:
            idx = self.seedance_resolution.findData(str(preferences["seedance_resolution"]))
            if idx >= 0:
                self.seedance_resolution.setCurrentIndex(idx)
        if "seedance_aspect_ratio" in preferences:
            idx = self.seedance_aspect_ratio.findData(str(preferences["seedance_aspect_ratio"]))
            if idx >= 0:
                self.seedance_aspect_ratio.setCurrentIndex(idx)
        if "seedance_duration_seconds" in preferences:
            idx = self.seedance_duration_seconds.findData(str(preferences["seedance_duration_seconds"]))
            if idx >= 0:
                self.seedance_duration_seconds.setCurrentIndex(idx)
        if "seedance_fps" in preferences:
            idx = self.seedance_fps.findData(str(preferences["seedance_fps"]))
            if idx >= 0:
                self.seedance_fps.setCurrentIndex(idx)
        if "seedance_motion_strength" in preferences:
            self.seedance_motion_strength.setValue(float(preferences["seedance_motion_strength"]))
        if "seedance_guidance_scale" in preferences:
            self.seedance_guidance_scale.setValue(float(preferences["seedance_guidance_scale"]))
        if "seedance_seed" in preferences:
            self.seedance_seed.setText(str(preferences["seedance_seed"]))
        if "seedance_negative_prompt" in preferences:
            self.seedance_negative_prompt.setText(str(preferences["seedance_negative_prompt"]))
        if "seedance_camera_motion" in preferences:
            idx = self.seedance_camera_motion.findData(str(preferences["seedance_camera_motion"]))
            if idx >= 0:
                self.seedance_camera_motion.setCurrentIndex(idx)
        if "seedance_style_preset" in preferences:
            idx = self.seedance_style_preset.findData(str(preferences["seedance_style_preset"]))
            if idx >= 0:
                self.seedance_style_preset.setCurrentIndex(idx)
        if "seedance_watermark" in preferences:
            self.seedance_watermark.setChecked(bool(preferences["seedance_watermark"]))
        if "seedance_extra_body" in preferences:
            self.seedance_extra_body.setPlainText(str(preferences["seedance_extra_body"]))
        if "stitch_crossfade_enabled" in preferences:
            self.stitch_crossfade_checkbox.setChecked(bool(preferences["stitch_crossfade_enabled"]))
        if "stitch_interpolation_enabled" in preferences:
            self.stitch_interpolation_checkbox.setChecked(bool(preferences["stitch_interpolation_enabled"]))
        if "stitch_interpolation_fps" in preferences:
            fps = str(preferences["stitch_interpolation_fps"]).strip()
            fps_index = self.stitch_interpolation_fps.findData(int(fps)) if fps.isdigit() else -1
            if fps_index >= 0:
                self.stitch_interpolation_fps.setCurrentIndex(fps_index)
        if "stitch_upscale_enabled" in preferences:
            self.stitch_upscale_checkbox.setChecked(bool(preferences["stitch_upscale_enabled"]))
        if "stitch_upscale_target" in preferences:
            target_index = self.stitch_upscale_target.findData(str(preferences["stitch_upscale_target"]))
            if target_index >= 0:
                self.stitch_upscale_target.setCurrentIndex(target_index)
        if "stitch_gpu_enabled" in preferences:
            self.stitch_gpu_checkbox.setChecked(bool(preferences["stitch_gpu_enabled"]))
        if "stitch_mute_original_audio" in preferences:
            self.stitch_mute_original_checkbox.setChecked(bool(preferences["stitch_mute_original_audio"]))
        if "stitch_original_audio_volume" in preferences:
            try:
                self.stitch_original_audio_volume.setValue(int(preferences["stitch_original_audio_volume"]))
            except (TypeError, ValueError):
                pass
        if "stitch_music_volume" in preferences:
            try:
                self.stitch_music_volume.setValue(int(preferences["stitch_music_volume"]))
            except (TypeError, ValueError):
                pass
        if "stitch_audio_fade_duration" in preferences:
            try:
                self.stitch_audio_fade_duration.setValue(float(preferences["stitch_audio_fade_duration"]))
            except (TypeError, ValueError):
                pass
        if "stitch_custom_music_file" in preferences:
            music_candidate = Path(str(preferences["stitch_custom_music_file"]))
            if str(preferences["stitch_custom_music_file"]).strip() and music_candidate.exists():
                self.custom_music_file = music_candidate
                self.music_file_label.setText(f"Music: {music_candidate.name}")
            else:
                self.custom_music_file = None
                self.music_file_label.setText("Music: none selected")
        if "crossfade_duration" in preferences:
            try:
                self.crossfade_duration.setValue(float(preferences["crossfade_duration"]))
            except (TypeError, ValueError):
                pass
        if "download_dir" in preferences:
            download_dir = Path(str(preferences["download_dir"]))
            try:
                download_dir.mkdir(parents=True, exist_ok=True)
                self.download_dir = download_dir
                self.download_path_input.setText(str(self.download_dir))
            except Exception:
                pass
        if "preview_muted" in preferences:
            self.preview_mute_checkbox.setChecked(bool(preferences["preview_muted"]))
        if "preview_volume" in preferences:
            try:
                self.preview_volume_slider.setValue(int(preferences["preview_volume"]))
            except (TypeError, ValueError):
                pass
        if "training_start_url" in preferences:
            self.training_start_url.setText(str(preferences["training_start_url"]))
        if "training_output_dir" in preferences:
            self.training_output_dir.setText(str(preferences["training_output_dir"]))
        if "training_timeout" in preferences:
            try:
                self.training_timeout.setValue(int(preferences["training_timeout"]))
            except (TypeError, ValueError):
                pass
        if "training_openai_model" in preferences:
            self.training_openai_model.setText(str(preferences["training_openai_model"]))
        if "training_use_embedded_browser" in preferences:
            self.training_use_embedded_browser.setChecked(bool(preferences["training_use_embedded_browser"]))
        if "training_trace_path" in preferences:
            self.training_trace_path.setText(str(preferences["training_trace_path"]))
        if "training_process_path" in preferences:
            self.training_process_path.setText(str(preferences["training_process_path"]))
        if "last_update_prompt_ts" in preferences:
            try:
                self.last_update_prompt_ts = int(preferences["last_update_prompt_ts"])
            except (TypeError, ValueError):
                self.last_update_prompt_ts = 0

        if "generated_videos" in preferences:
            self._restore_video_list_from_preferences(
                preferences.get("generated_videos"),
                selected_index=preferences.get("selected_generated_video_index"),
            )

        self._toggle_prompt_source_fields()
        self._sync_video_options_label()
        self._applying_preferences = False

    def _save_preferences_to_path(
        self,
        file_path: Path,
        *,
        show_feedback: bool = False,
        only_if_changed: bool = False,
    ) -> bool:
        try:
            preferences = self._collect_preferences()
            current_signature = self._preferences_signature(preferences)
            if only_if_changed and current_signature == self._last_saved_preferences_signature:
                return False
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(preferences, handle, indent=2)
        except Exception as exc:
            if show_feedback:
                QMessageBox.critical(self, "Save Preferences Failed", str(exc))
            self._append_log(f"ERROR: Could not save preferences: {exc}")
            return False

        self._last_saved_preferences_signature = current_signature
        self._append_log(f"Saved preferences to: {file_path}")
        return True

    def _load_preferences_from_path(self, file_path: Path, *, show_feedback: bool = False) -> bool:
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                preferences = json.load(handle)
            self._apply_preferences(preferences)
            if isinstance(preferences, dict):
                self._last_saved_preferences_signature = self._preferences_signature(preferences)
        except Exception as exc:
            if show_feedback:
                QMessageBox.critical(self, "Load Preferences Failed", str(exc))
            self._append_log(f"ERROR: Could not load preferences: {exc}")
            return False

        self._append_log(f"Loaded preferences from: {file_path}")
        return True

    def save_model_api_settings(self) -> None:
        if self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=True):
            if self.qtwebengine_remote_debug_enabled.isChecked():
                os.environ["GROK_QTWEBENGINE_REMOTE_DEBUG_PORT"] = str(self.qtwebengine_remote_debug_port.value())
            else:
                os.environ.pop("GROK_QTWEBENGINE_REMOTE_DEBUG_PORT", None)
            self._reset_cdp_relay_session_state()
            QMessageBox.information(
                self,
                "Settings Saved",
                f"Settings saved to:\n{DEFAULT_PREFERENCES_FILE}\n\n"
                "QtWebEngine remote debugging changes apply after restarting the app.",
            )

    def _reset_cdp_relay_session_state(self) -> None:
        self._cdp_relay_temporarily_disabled = False

    def _load_startup_preferences(self) -> None:
        self._load_preferences_from_path(DEFAULT_PREFERENCES_FILE, show_feedback=False)

    def save_preferences(self) -> None:
        self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=True)

    def load_preferences(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preferences",
            str(DEFAULT_PREFERENCES_FILE),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        self._load_preferences_from_path(Path(file_path), show_feedback=True)

    def _append_automation_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "automation_log") and self.automation_log is not None:
            self.automation_log.append(f"[{timestamp}] {text}")
        self._append_log(f"Automation: {text}")

    def _ensure_automation_runtime(self) -> AutomationRuntimeWorker:
        if self.automation_runtime is None:
            runtime = AutomationRuntimeWorker(BASE_DIR / "extension")
            runtime.log.connect(self._append_automation_log)
            runtime.status.connect(lambda k, v: self._append_automation_log(f"{k}: {v}"))
            runtime.start()
            self.automation_runtime = runtime
            time.sleep(0.2)
        return self.automation_runtime

    def _start_automation_chrome(self) -> None:
        try:
            runtime = self._ensure_automation_runtime()
            instance = runtime.start_chrome()
            self.automation_chrome_instance = instance
            self._append_automation_log(f"Chrome websocket endpoint: {instance.ws_endpoint}")
        except Exception as exc:
            self._append_automation_log(f"Failed to start Automation Chrome: {exc}")

    def _connect_automation_cdp(self) -> None:
        try:
            runtime = self._ensure_automation_runtime()
            title = runtime.connect_cdp()
            self._append_automation_log(f"CDP connected: smoke test page title '{title}'")
        except Exception as exc:
            self._append_automation_log(f"CDP connection failed: {exc}")
            QMessageBox.warning(self, "CDP connection failed", f"Could not connect over CDP.\n\n{exc}")

    def _run_extension_dom_ping(self) -> None:
        try:
            runtime = self._ensure_automation_runtime()
            result = runtime.dom_ping()
            self._append_automation_log(f"Extension DOM ping ack: {json.dumps(result, ensure_ascii=False)}")
        except Exception as exc:
            self._append_automation_log(f"Extension DOM ping failed: {exc}")

    def _ensure_udp_service(self) -> None:
        runtime = self._ensure_automation_runtime()
        runtime.ensure_udp_service()

    def _run_social_upload_via_mode(self, platform_name: str, video_path: str, caption: str, title: str) -> None:
        mode = str(self.automation_mode.currentData() if hasattr(self, "automation_mode") else "embedded")
        if mode != "udp":
            self._start_social_browser_upload(platform_name=platform_name, video_path=video_path, caption=caption, title=title)
            return

        self._cancel_social_upload_run(platform_name, reason="switching to UDP automation")
        self._ensure_udp_service()
        if self.udp_workflow_worker is not None and self.udp_workflow_worker.isRunning():
            self._append_automation_log("Stopping previous UDP workflow before starting a new one.")
            try:
                self.udp_workflow_worker.request_stop()
                if not self.udp_workflow_worker.wait(2000):
                    self.udp_workflow_worker.requestInterruption()
                    if not self.udp_workflow_worker.wait(1500):
                        self._append_automation_log(
                            "WARNING: Previous UDP workflow did not stop in time; forcing thread termination."
                        )
                        self.udp_workflow_worker.terminate()
                        self.udp_workflow_worker.wait(800)
            except Exception as exc:
                self._append_automation_log(f"WARNING: Failed to stop previous UDP workflow cleanly: {exc}")
        self._append_automation_log("UDP action log file: logs/udp_automation.log")
        self._append_automation_log(f"Starting UDP workflow for {platform_name}.")
        worker = UdpWorkflowWorker(platform_name=platform_name, video_path=video_path, title=title, caption=caption)
        worker.finished_with_result.connect(lambda result: self._append_automation_log(f"{platform_name} UDP result: {result}"))
        worker.failed.connect(lambda err: self._append_automation_log(f"{platform_name} UDP failed: {err}"))
        worker.finished.connect(lambda: setattr(self, "udp_workflow_worker", None))
        self.udp_workflow_worker = worker
        worker.start()

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.appendPlainText(f"[{timestamp}] {text}")

    def clear_activity_log(self) -> None:
        self.log.clear()
        self._append_log("Activity log cleared.")

    def jump_activity_log_to_bottom(self) -> None:
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_browser_load_finished(self, ok: bool) -> None:
        if ok:
            self._playback_hack_success_logged = False
            self.video_playback_hack_timer.start()
            self._ensure_browser_video_playback()
            if self.continue_from_frame_waiting_for_reload and self.continue_from_frame_active:
                self.continue_from_frame_waiting_for_reload = False
                self.continue_from_frame_reload_timeout_timer.stop()
                self._append_log(
                    "Continue-from-last-frame: detected page reload after image upload. Proceeding with prompt entry."
                )
                QTimer.singleShot(700, lambda: self._start_manual_browser_generation(self.continue_from_frame_prompt, 1))
            if self.embedded_training_active and self.training_use_embedded_browser.isChecked():
                self._inject_embedded_training_capture_script()

    def _retry_continue_after_small_download(self, variant: int) -> None:
        source_video = self.continue_from_frame_current_source_video
        self._append_log(
            f"Variant {variant} download is smaller than 1MB; restarting from previous source video: {source_video}"
        )

        if not source_video or not Path(source_video).exists():
            self._append_log(
                "ERROR: Previous source video is unavailable for retry; continue-from-last-frame workflow is stopping."
            )
            self.continue_from_frame_active = False
            self.continue_from_frame_target_count = 0
            self.continue_from_frame_completed = 0
            self.continue_from_frame_prompt = ""
            self.continue_from_frame_current_source_video = ""
            return

        frame_path = self._extract_last_frame(source_video)
        if frame_path is None:
            self._append_log(
                "ERROR: Could not extract a last frame from the previous source video; "
                "continue-from-last-frame workflow is stopping."
            )
            self.continue_from_frame_active = False
            self.continue_from_frame_target_count = 0
            self.continue_from_frame_completed = 0
            self.continue_from_frame_prompt = ""
            self.continue_from_frame_current_source_video = ""
            return

        self.last_extracted_frame_path = frame_path
        self._upload_frame_into_grok(frame_path, on_uploaded=self._wait_for_continue_upload_reload)

    def _ensure_browser_video_playback(self) -> None:
        if not hasattr(self, "browser") or self.browser is None:
            return

        script = r"""
            (() => {
                try {
                    const videos = [...document.querySelectorAll("video")];
                    if (!videos.length) return { ok: true, found: 0, attempted: 0, playing: 0 };

                    const pokeUserGesture = () => {
                        try {
                            const ev = new MouseEvent("click", { bubbles: true, cancelable: true, composed: true });
                            document.body.dispatchEvent(ev);
                        } catch (_) {}
                    };
                    pokeUserGesture();

                    const common = { bubbles: true, cancelable: true, composed: true };
                    const synthClick = (el) => {
                        if (!el) return;
                        try {
                            el.dispatchEvent(new PointerEvent("pointerdown", common));
                            el.dispatchEvent(new MouseEvent("mousedown", common));
                            el.dispatchEvent(new PointerEvent("pointerup", common));
                            el.dispatchEvent(new MouseEvent("mouseup", common));
                            el.dispatchEvent(new MouseEvent("click", common));
                        } catch (_) {}
                    };

                    let attempted = 0;
                    let playing = 0;
                    for (const video of videos) {
                        try {
                            // Chromium/WebEngine autoplay policies are stricter for unmuted media.
                            // Start muted so playback can begin reliably, then attempt an unmute.
                            video.muted = true;
                            video.defaultMuted = true;
                            video.volume = Math.max(video.volume || 0, 0.01);
                            video.autoplay = true;
                            video.playsInline = true;
                            video.loop = false;
                            video.setAttribute("muted", "");
                            video.setAttribute("autoplay", "");
                            video.setAttribute("playsinline", "");
                            video.setAttribute("webkit-playsinline", "");
                            video.controls = true;
                            video.preload = "auto";

                            const st = getComputedStyle(video);
                            if (st.display === "none") video.style.display = "block";
                            if (st.visibility === "hidden") video.style.visibility = "visible";
                            if (Number(st.opacity || "1") < 0.1) video.style.opacity = "1";

                            if (video.paused || video.readyState < 2) {
                                attempted += 1;
                                const p = video.play();
                                if (p && typeof p.catch === "function") {
                                    p.catch(() => {
                                        const playButton = video.closest("[role='button'], button")
                                            || video.parentElement?.querySelector("button, [role='button']");
                                        synthClick(playButton);
                                        const p2 = video.play();
                                        if (p2 && typeof p2.catch === "function") p2.catch(() => {});
                                    });
                                    p.then(() => {
                                        // Best-effort unmute; if policy rejects it we keep muted playback.
                                        try {
                                            video.muted = false;
                                            video.defaultMuted = false;
                                            video.removeAttribute("muted");
                                            video.volume = Math.max(video.volume || 0, 0.2);
                                        } catch (_) {}
                                    });
                                }

                                if (video.muted) {
                                    // Keep muted for autoplay reliability.
                                }

                                if (video.readyState < 2) {
                                    video.addEventListener("canplay", () => {
                                        const p3 = video.play();
                                        if (p3 && typeof p3.catch === "function") p3.catch(() => {});
                                    }, { once: true });
                                }
                            }

                            if (!video.paused && !video.ended && video.currentTime >= 0) {
                                playing += 1;
                            }
                        } catch (_) {}
                    }

                    return { ok: true, found: videos.length, attempted, playing };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        def after(result):
            if not isinstance(result, dict) or not result.get("ok"):
                return
            if result.get("found", 0) > 0 and result.get("playing", 0) > 0 and not self._playback_hack_success_logged:
                self._append_log(
                    f"Playback hack active: {result.get('playing')}/{result.get('found')} embedded video element(s) are playing."
                )
                self._playback_hack_success_logged = True

        self.browser.page().runJavaScript(script, after)

    def _aspect_ratio_from_video_size(self, size: str) -> str:
        mapping = {
            "720x1280": "9:16",
            "1280x720": "16:9",
            "1024x1792": "4:7",
            "1792x1024": "7:4",
        }
        return mapping.get(size, str(self.video_aspect_ratio.currentData() or "16:9"))

    def _confirm_openai_sora_settings(self, sora_settings: dict[str, object]) -> bool:
        model = str(sora_settings.get("model") or "sora-2")
        size = str(sora_settings.get("size") or "1280x720")
        seconds = str(sora_settings.get("seconds") or "8")
        input_reference = str(sora_settings.get("input_reference") or "").strip()
        continue_from_last_frame = str(sora_settings.get("continue_from_last_frame") or "").strip().lower() in {"1", "true", "yes", "on"}
        extra_body = sora_settings.get("extra_body") if isinstance(sora_settings.get("extra_body"), dict) else {}
        aspect_ratio = self._aspect_ratio_from_video_size(size)
        message = (
            "OpenAI Sora request will be sent with:\n\n"
            f"• model: {model}\n"
            "• prompt: <generated prompt>\n"
            f"• size: {size}\n"
            f"• seconds: {seconds}\n"
            f"• aspect ratio target: {aspect_ratio}\n"
            f"• input_reference: {input_reference or '(none)'}\n"
            f"• continue_from_last_frame: {'yes' if continue_from_last_frame else 'no'}\n"
            f"• extra_body fields: {len(extra_body)}\n\n"
            "Allowed values (from current OpenAI SDK docs):\n"
            "• seconds: 4, 8, 12\n"
            "• size: 720x1280, 1280x720, 1024x1792, 1792x1024\n\n"
            "Continue with generation?"
        )
        answer = QMessageBox.question(
            self,
            "OpenAI Sora Parameters",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_seedance_settings(self, seedance_settings: dict[str, object]) -> bool:
        model = str(seedance_settings.get("model") or "seedance-2.0")
        resolution = str(seedance_settings.get("resolution") or "1280x720")
        aspect_ratio = str(seedance_settings.get("aspect_ratio") or "16:9")
        duration_seconds = str(seedance_settings.get("duration_seconds") or "8")
        fps = str(seedance_settings.get("fps") or "24")
        motion_strength = str(seedance_settings.get("motion_strength") or "0.6")
        guidance_scale = str(seedance_settings.get("guidance_scale") or "7.5")
        camera_motion = str(seedance_settings.get("camera_motion") or "(default)")
        style_preset = str(seedance_settings.get("style_preset") or "(default)")
        use_watermark = "yes" if bool(seedance_settings.get("watermark")) else "no"
        message = (
            "Seedance 2.0 request will be sent with\n\n"
            f"• model: {model}\n"
            "• prompt: <concept/manual prompt from app>\n"
            f"• resolution: {resolution}\n"
            f"• aspect_ratio: {aspect_ratio}\n"
            f"• duration_seconds: {duration_seconds}\n"
            f"• fps: {fps}\n"
            f"• motion_strength: {motion_strength}\n"
            f"• guidance_scale: {guidance_scale}\n"
            f"• camera_motion: {camera_motion}\n"
            f"• style_preset: {style_preset}\n"
            f"• watermark: {use_watermark}\n"
            f"• extra_body fields: {len(seedance_settings.get('extra_body', {}))}\n\n"
            "Continue with generation?"
        )
        answer = QMessageBox.question(
            self,
            "Seedance 2.0 Parameters",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def start_generation(self) -> None:
        self.stop_all_requested = False
        concept = self.concept.toPlainText().strip()
        prompt_source = self.prompt_source.currentData()
        video_provider = self.video_provider.currentData()
        manual_prompt = self.manual_prompt.toPlainText().strip()
        api_key = self.api_key.text().strip()

        if prompt_source != "manual" and not concept:
            QMessageBox.warning(self, "Missing Concept", "Please enter a concept.")
            return
        if prompt_source == "manual" and not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return
        if prompt_source == "openai" and not (self.openai_api_key.text().strip() or self.openai_access_token.text().strip()):
            QMessageBox.warning(
                self,
                "Missing OpenAI Credentials",
                "Please provide an OpenAI API key or authorize OpenAI in browser (or paste an access token).",
            )
            return

        if video_provider == "grok" and not api_key:
            QMessageBox.warning(self, "Missing Grok API Key", "Please enter a Grok API key for Grok Imagine video generation.")
            return

        if video_provider == "openai" and not (self.openai_api_key.text().strip() or self.openai_access_token.text().strip()):
            QMessageBox.warning(
                self,
                "Missing OpenAI Credentials",
                "Please provide an OpenAI API key or access token for Sora video generation.",
            )
            return

        if video_provider == "seedance" and not (self.seedance_api_key.text().strip() or self.seedance_oauth_token.text().strip()):
            QMessageBox.warning(
                self,
                "Missing Seedance Credentials",
                "Please provide a Seedance API key or OAuth token in Model/API Settings.",
            )
            return

        config = GrokConfig(
            api_key=api_key,
            chat_model=self.chat_model.text().strip() or "grok-3-mini",
            image_model=self.image_model.text().strip() or "grok-video-latest",
        )

        selected_resolution = str(self.video_resolution.currentData() or "1280x720")
        selected_resolution_label = self.video_resolution.currentText().split(" ", 1)[0]
        selected_duration_seconds = int(self.video_duration.currentData() or 10)
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")
        openai_sora_settings: dict[str, object] = {}
        seedance_settings: dict[str, object] = {}

        if video_provider == "openai":
            try:
                openai_sora_settings = self._collect_sora2_settings()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Sora 2 Settings", str(exc))
                return

            selected_resolution = str(openai_sora_settings.get("size") or selected_resolution)
            selected_resolution_label = selected_resolution
            selected_aspect_ratio = self._aspect_ratio_from_video_size(selected_resolution)
            seconds_value = str(openai_sora_settings.get("seconds") or "8")
            try:
                selected_duration_seconds = int(seconds_value)
            except ValueError:
                QMessageBox.warning(self, "Invalid Sora 2 Settings", "seconds must be one of 4, 8, or 12.")
                return

        if video_provider == "openai" and self._openai_sora_help_always_show:
            if not self._confirm_openai_sora_settings(openai_sora_settings):
                self._append_log("OpenAI generation canceled from Sora parameters confirmation dialog.")
                return

        if video_provider == "seedance":
            try:
                seedance_settings = self._collect_seedance_settings()
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Seedance Settings", str(exc))
                return

            selected_resolution = str(seedance_settings.get("resolution") or selected_resolution)
            selected_resolution_label = selected_resolution
            selected_aspect_ratio = str(seedance_settings.get("aspect_ratio") or selected_aspect_ratio)
            selected_duration_seconds = int(seedance_settings.get("duration_seconds") or selected_duration_seconds)

            if not self._confirm_seedance_settings(seedance_settings):
                self._append_log("Seedance generation canceled from parameter confirmation dialog.")
                return

        prompt_config = PromptConfig(
            prompt_source=prompt_source,
            video_provider=video_provider,
            concept=concept,
            manual_prompt=manual_prompt,
            openai_api_key=self.openai_api_key.text().strip(),
            openai_access_token=self.openai_access_token.text().strip(),
            openai_chat_model=self.openai_chat_model.text().strip() or "gpt-5.1-codex",
            video_resolution=selected_resolution,
            video_resolution_label=selected_resolution_label,
            video_aspect_ratio=selected_aspect_ratio,
            video_duration_seconds=selected_duration_seconds,
            openai_sora_settings=openai_sora_settings,
            seedance_settings=seedance_settings,
        )

        self.worker = GenerateWorker(config, prompt_config, self.count.value(), self.download_dir)
        self.worker.status.connect(self._append_log)
        self.worker.finished_video.connect(self.on_video_finished)
        self.worker.failed.connect(self.on_generation_error)
        self.worker.start()

    def _openai_pkce_challenge(self, verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")

    def _tiktok_pkce_challenge(self, verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        if TIKTOK_PKCE_CHALLENGE_ENCODING == "base64url":
            return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        return digest.hex()

    def _generate_pkce_verifier(self, length: int = 64) -> str:
        length = max(43, min(128, int(length)))
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def _start_oauth_callback_listener(self, port: int):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        result: dict[str, str] = {}
        event = threading.Event()

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # type: ignore[override]
                parsed = urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not found")
                    return

                params = _parse_query_preserving_plus(parsed.query)
                for key in ("code", "state", "error", "error_description"):
                    value = str(params.get(key, ""))
                    if value:
                        result[key] = value

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authorization received.</h2><p>You can close this tab and return to the app.</p></body></html>"
                )
                event.set()

            def log_message(self, format, *args):  # type: ignore[override]
                return

        server = ThreadingHTTPServer(("127.0.0.1", port), CallbackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, event, result

    def _exchange_openai_oauth_code(self, code: str, redirect_uri: str, code_verifier: str) -> dict:
        request_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "code_verifier": code_verifier,
        }
        last_error: str | None = None

        for token_path in OPENAI_TOKEN_PATHS:
            token_endpoint = f"{OPENAI_OAUTH_ISSUER}{token_path}"
            response = requests.post(token_endpoint, data=request_data, timeout=60)
            if response.ok:
                payload = response.json()
                if not payload.get("access_token"):
                    raise RuntimeError("OAuth token response did not include access_token.")
                return payload
            last_error = f"{response.status_code} {response.text[:500]}"

        raise RuntimeError(f"OAuth token exchange failed for all configured endpoints: {last_error}")

    def _run_openai_oauth_flow(self) -> None:
        state = secrets.token_hex(16)
        verifier = self._generate_pkce_verifier(64)
        challenge = self._openai_pkce_challenge(verifier)
        redirect_uri = f"http://localhost:{OPENAI_OAUTH_CALLBACK_PORT}/auth/callback"

        server, done_event, callback_result = self._start_oauth_callback_listener(OPENAI_OAUTH_CALLBACK_PORT)
        try:
            query = urlencode(
                {
                    "response_type": "code",
                    "client_id": OPENAI_CODEX_CLIENT_ID,
                    "redirect_uri": redirect_uri,
                    "scope": OPENAI_OAUTH_SCOPE,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": state,
                    "id_token_add_organizations": "true",
                }
            )
            authorize_url = f"{OPENAI_OAUTH_ISSUER}/oauth/authorize?{query}&codex_cli_simplified_flow"

            opened = QDesktopServices.openUrl(QUrl(authorize_url))
            if opened:
                self._append_log("Opened OpenAI OAuth authorize URL in your system browser. Complete sign-in to continue.")
            else:
                self._append_log("Could not launch system browser for OAuth authorize URL.")
                raise RuntimeError("Failed to open system browser for OpenAI OAuth authorization.")

            timeout_s = 240
            start = time.time()
            while not done_event.is_set() and (time.time() - start) < timeout_s:
                QApplication.processEvents()
                time.sleep(0.1)

            if not done_event.is_set():
                raise TimeoutError("Timed out waiting for OpenAI OAuth callback.")

            if callback_result.get("error"):
                desc = callback_result.get("error_description") or callback_result["error"]
                raise RuntimeError(f"OpenAI OAuth authorization failed: {desc}")

            callback_state = callback_result.get("state", "")
            if callback_state != state:
                raise RuntimeError("OpenAI OAuth state mismatch; please retry authorization.")

            code = callback_result.get("code", "")
            if not code:
                raise RuntimeError("OpenAI OAuth callback did not include an authorization code.")

            token_payload = self._exchange_openai_oauth_code(code, redirect_uri, verifier)
            access_token = str(token_payload.get("access_token", "")).strip()
            refresh_token = str(token_payload.get("refresh_token", "")).strip()
            self.openai_access_token.setText(access_token)
            self._append_log("OpenAI OAuth complete. Access token has been populated in OpenAI Access Token.")
            if refresh_token:
                self._append_log("Refresh token received (not persisted yet). Re-run browser authorization if token expires.")
        finally:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass

    def _exchange_facebook_oauth_code(self, code: str, redirect_uri: str, app_id: str, app_secret: str) -> str:
        response = requests.get(
            FACEBOOK_OAUTH_TOKEN_URL,
            params={
                "client_id": app_id,
                "client_secret": app_secret,
                "redirect_uri": redirect_uri,
                "code": code,
            },
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Facebook OAuth token exchange failed: {response.status_code} {response.text[:500]}")
        payload = response.json()
        access_token = str(payload.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError("Facebook OAuth token response did not include access_token.")
        return access_token

    def _fetch_facebook_pages_for_user(self, user_access_token: str) -> list[dict]:
        response = requests.get(
            f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/me/accounts",
            params={"access_token": user_access_token, "fields": "id,name,access_token"},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"Failed to fetch Facebook Pages: {response.status_code} {response.text[:500]}")
        payload = response.json()
        pages = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(pages, list):
            raise RuntimeError(f"Unexpected Facebook Pages response: {payload}")
        return [page for page in pages if isinstance(page, dict)]

    def _run_facebook_oauth_flow(self) -> None:
        app_id = self.facebook_app_id.text().strip()
        app_secret = self.facebook_app_secret.text().strip()
        if not app_id:
            raise RuntimeError("Facebook App ID is required for OAuth.")
        if not app_secret:
            raise RuntimeError("Facebook App Secret is required for OAuth.")

        state = secrets.token_hex(16)
        redirect_uri = f"http://localhost:{FACEBOOK_OAUTH_CALLBACK_PORT}/auth/callback"
        server, done_event, callback_result = self._start_oauth_callback_listener(FACEBOOK_OAUTH_CALLBACK_PORT)

        try:
            authorize_url = (
                f"{FACEBOOK_OAUTH_AUTHORIZE_URL}?"
                f"{urlencode({'client_id': app_id, 'redirect_uri': redirect_uri, 'state': state, 'response_type': 'code', 'scope': FACEBOOK_OAUTH_SCOPE})}"
            )

            opened = QDesktopServices.openUrl(QUrl(authorize_url))
            if opened:
                self._append_log("Opened Facebook OAuth authorize URL in your system browser. Complete sign-in to continue.")
            else:
                raise RuntimeError("Failed to open system browser for Facebook OAuth authorization.")

            timeout_s = 240
            start = time.time()
            while not done_event.is_set() and (time.time() - start) < timeout_s:
                QApplication.processEvents()
                time.sleep(0.1)

            if not done_event.is_set():
                raise TimeoutError("Timed out waiting for Facebook OAuth callback.")

            if callback_result.get("error"):
                desc = callback_result.get("error_description") or callback_result["error"]
                raise RuntimeError(f"Facebook OAuth authorization failed: {desc}")

            callback_state = callback_result.get("state", "")
            if callback_state != state:
                raise RuntimeError("Facebook OAuth state mismatch; please retry authorization.")

            code = callback_result.get("code", "")
            if not code:
                raise RuntimeError("Facebook OAuth callback did not include an authorization code.")

            user_access_token = self._exchange_facebook_oauth_code(code, redirect_uri, app_id, app_secret)
            pages = self._fetch_facebook_pages_for_user(user_access_token)
            if not pages:
                raise RuntimeError("Facebook OAuth succeeded, but no Pages were returned for this account.")

            preferred_page_id = self.facebook_page_id.text().strip()
            selected_page = None
            if preferred_page_id:
                selected_page = next((page for page in pages if str(page.get("id")) == preferred_page_id), None)
            if selected_page is None:
                selected_page = pages[0]

            page_id = str(selected_page.get("id") or "").strip()
            page_name = str(selected_page.get("name") or "").strip() or page_id
            page_access_token = str(selected_page.get("access_token") or "").strip()
            if not page_id or not page_access_token:
                raise RuntimeError("Selected Facebook Page did not include required id/access_token.")

            self.facebook_page_id.setText(page_id)
            self.facebook_access_token.setText(page_access_token)
            self._append_log(f"Facebook OAuth complete. Using Page '{page_name}' ({page_id}). Token populated.")

            if preferred_page_id and preferred_page_id != page_id:
                self._append_log(
                    f"Requested Page ID {preferred_page_id} was not found for this user; selected first available Page {page_id}."
                )
        finally:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass

    def authorize_facebook_pages(self) -> None:
        try:
            self._run_facebook_oauth_flow()
        except Exception as exc:
            self._append_log(f"ERROR: Facebook OAuth flow failed: {exc}")
            QMessageBox.critical(self, "Facebook OAuth Failed", str(exc))

    def _exchange_tiktok_oauth_code(
        self,
        code: str,
        redirect_uri: str,
        client_key: str,
        client_secret: str,
        code_verifier: str,
    ) -> str:
        response = requests.post(
            TIKTOK_OAUTH_TOKEN_URL,
            data={
                "client_key": client_key,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"TikTok OAuth token exchange failed: {response.status_code} {response.text[:500]}")

        payload = response.json()
        token_payload = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(token_payload, dict):
            token_payload = payload if isinstance(payload, dict) else {}

        access_token = str(token_payload.get("access_token") or "").strip()
        if not access_token:
            raise RuntimeError(f"TikTok OAuth token response did not include access_token: {payload}")
        return access_token

    def _run_tiktok_oauth_flow(self) -> None:
        client_key = self.tiktok_client_key.text().strip()
        client_secret = self.tiktok_client_secret.text().strip()
        if not client_key:
            raise RuntimeError("TikTok Client Key is required for OAuth.")
        if not client_secret:
            raise RuntimeError("TikTok Client Secret is required for OAuth.")

        state = secrets.token_hex(16)
        verifier = self._generate_pkce_verifier(64)
        challenge = self._tiktok_pkce_challenge(verifier)
        redirect_uri = f"http://localhost:{TIKTOK_OAUTH_CALLBACK_PORT}/auth/callback"
        server, done_event, callback_result = self._start_oauth_callback_listener(TIKTOK_OAUTH_CALLBACK_PORT)

        try:
            authorize_url = (
                f"{TIKTOK_OAUTH_AUTHORIZE_URL}?"
                f"{urlencode({'client_key': client_key, 'redirect_uri': redirect_uri, 'state': state, 'response_type': 'code', 'scope': TIKTOK_OAUTH_SCOPE, 'code_challenge': challenge, 'code_challenge_method': 'S256'})}"
            )

            opened = QDesktopServices.openUrl(QUrl(authorize_url))
            if opened:
                self._append_log("Opened TikTok OAuth authorize URL in your system browser. Complete sign-in to continue.")
            else:
                raise RuntimeError("Failed to open system browser for TikTok OAuth authorization.")

            timeout_s = 240
            start = time.time()
            while not done_event.is_set() and (time.time() - start) < timeout_s:
                QApplication.processEvents()
                time.sleep(0.1)

            if not done_event.is_set():
                raise TimeoutError("Timed out waiting for TikTok OAuth callback.")

            if callback_result.get("error"):
                desc = callback_result.get("error_description") or callback_result["error"]
                raise RuntimeError(f"TikTok OAuth authorization failed: {desc}")

            callback_state = callback_result.get("state", "")
            if callback_state != state:
                raise RuntimeError("TikTok OAuth state mismatch; please retry authorization.")

            code = callback_result.get("code", "")
            if not code:
                raise RuntimeError("TikTok OAuth callback did not include an authorization code.")

            access_token = self._exchange_tiktok_oauth_code(code, redirect_uri, client_key, client_secret, verifier)
            self.tiktok_access_token.setText(access_token)
            self._save_preferences_to_path(DEFAULT_PREFERENCES_FILE, show_feedback=False)
            self._append_log(
                "TikTok OAuth complete. Access token has been populated in TikTok Access Token and saved to preferences.json."
            )
            QMessageBox.information(
                self,
                "TikTok OAuth Complete",
                "TikTok access token received and stored in settings. You can upload now.",
            )
        finally:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass

    def authorize_tiktok_upload(self) -> None:
        try:
            self._run_tiktok_oauth_flow()
        except Exception as exc:
            self._append_log(f"ERROR: TikTok OAuth flow failed: {exc}")
            QMessageBox.critical(self, "TikTok OAuth Failed", str(exc))

    def open_ai_provider_login(self) -> None:
        prompt_source = self.prompt_source.currentData()
        video_provider = self.video_provider.currentData()
        if prompt_source == "openai" or video_provider == "openai":
            try:
                self._run_openai_oauth_flow()
            except Exception as exc:
                self._append_log(f"ERROR: OpenAI OAuth flow failed: {exc}")
            return
        if video_provider == "seedance":
            QDesktopServices.openUrl(QUrl("https://seedance.ai/"))
            self._append_log("Opened Seedance in browser. If OAuth is supported on your account, complete sign-in and paste token.")
            return

        self.browser.setUrl(QUrl("https://grok.com/"))
        self._append_log("Opened Grok in browser for sign-in.")

    def _describe_overlay_frame_with_selected_ai(self, frame_path: Path, timestamp_seconds: float) -> str:
        source = self.prompt_source.currentData()
        instruction = (
            "Analyze this frame image and return one short subtitle line (max 12 words, no hashtags, no quotes). "
            f"Timestamp in source video: {timestamp_seconds:.2f}s. "
            "If uncertain, provide a neutral literal scene description."
        )

        if source == "openai":
            openai_credential = self.openai_api_key.text().strip() or self.openai_access_token.text().strip()
            if not openai_credential:
                raise RuntimeError("OpenAI API key or access token is required.")

            model = self.openai_chat_model.text().strip() or "gpt-5.1-codex"
            image_bytes = frame_path.read_bytes()
            data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"

            file_id = ""
            use_uploaded_file = _openai_is_likely_api_key(openai_credential)
            if use_uploaded_file:
                upload_headers = _openai_headers_from_credential(openai_credential)
                upload_headers.pop("Content-Type", None)
                with frame_path.open("rb") as image_handle:
                    files_payload = {
                        "file": (frame_path.name, image_handle, "image/png"),
                    }
                    upload_response = requests.post(
                        f"{OPENAI_API_BASE}/files",
                        headers=upload_headers,
                        data={"purpose": "vision"},
                        files=files_payload,
                        timeout=120,
                    )
                if not upload_response.ok:
                    if upload_response.status_code == 401 and "must be made with a secret key" in upload_response.text:
                        use_uploaded_file = False
                    else:
                        raise RuntimeError(
                            f"OpenAI file upload failed: {upload_response.status_code} {upload_response.text[:400]}"
                        )

                if use_uploaded_file:
                    try:
                        upload_payload = upload_response.json() if upload_response.content else {}
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"OpenAI file upload returned non-JSON response ({exc}): {upload_response.text[:300]}"
                        ) from exc

                    file_id = str((upload_payload or {}).get("id") or "").strip()
                    if not file_id:
                        raise RuntimeError("OpenAI file upload did not return a file id.")

            try:
                response_headers = _openai_headers_from_credential(openai_credential)
                image_part: dict[str, str]
                if use_uploaded_file and file_id:
                    image_part = {"type": "input_image", "file_id": file_id}
                else:
                    image_part = {"type": "input_image", "image_url": data_url}

                payload = {
                    "model": model,
                    "instructions": "You describe video frames with concise subtitle-style captions.",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": instruction},
                                image_part,
                            ],
                        }
                    ],
                    "store": False,
                }
                response = requests.post(
                    f"{OPENAI_API_BASE}/responses",
                    headers=response_headers,
                    json=payload,
                    timeout=120,
                )
                if not response.ok:
                    raise RuntimeError(f"OpenAI vision request failed: {response.status_code} {response.text[:400]}")

                streamed_text = _extract_text_from_responses_sse(response.text)
                if streamed_text:
                    return streamed_text.strip()

                try:
                    response_payload = response.json() if response.content else {}
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"OpenAI vision returned non-JSON response ({exc}): {response.text[:300]}"
                    ) from exc

                text = _extract_text_from_responses_body(response_payload)
                if text:
                    return text.strip()

                raise RuntimeError("OpenAI vision response did not include text output.")
            finally:
                if file_id:
                    try:
                        requests.delete(
                            f"{OPENAI_API_BASE}/files/{file_id}",
                            headers=_openai_headers_from_credential(openai_credential),
                            timeout=30,
                        )
                    except Exception:
                        pass

        image_bytes = frame_path.read_bytes()
        data_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        return self._call_selected_ai(
            "You describe video frames with concise subtitle-style captions.",
            f"{instruction}\n\nFrame image (data URL): {data_url}",
        )

    def _describe_overlay_prompt_with_selected_ai(self, prompt_text: str, timestamp_seconds: float) -> str:
        cleaned_prompt = (prompt_text or "").strip()
        if not cleaned_prompt:
            raise RuntimeError("Prompt text is required for prompt-only AI captioning.")

        return self._call_selected_ai(
            "You write concise subtitle lines for videos.",
            (
                "Generate one short subtitle line (max 12 words, no hashtags, no quotes) based on the prompt context. "
                f"Timestamp in source video: {timestamp_seconds:.2f}s. "
                "Vary phrasing naturally across timestamps while staying consistent with the prompt context.\n\n"
                f"Prompt context: {cleaned_prompt}"
            ),
        )

    def _call_selected_ai(self, system: str, user: str) -> str:
        source = self.prompt_source.currentData()
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }

        if source == "openai":
            openai_credential = self.openai_api_key.text().strip() or self.openai_access_token.text().strip()
            if not openai_credential:
                raise RuntimeError("OpenAI API key or access token is required.")
            return _call_openai_chat_api(
                credential=openai_credential,
                model=self.openai_chat_model.text().strip() or "gpt-5.1-codex",
                system=system,
                user=user,
                temperature=0.4,
            )

        grok_key = self.api_key.text().strip()
        if not grok_key:
            if self.ai_auth_method.currentData() == "browser":
                raise RuntimeError("Browser authorization selected, but API-based Grok requests still require a GROK API key.")
            raise RuntimeError("Grok API key is required.")
        headers["Authorization"] = f"Bearer {grok_key}"
        payload["model"] = self.chat_model.text().strip() or "grok-3-mini"
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=90)
        if not response.ok:
            raise RuntimeError(f"Grok request failed: {response.status_code} {response.text[:400]}")
        return response.json()["choices"][0]["message"]["content"].strip()

    def generate_prompt_from_concept(self) -> None:
        concept = self.concept.toPlainText().strip()
        source = self.prompt_source.currentData()
        if source not in {"grok", "openai"}:
            QMessageBox.warning(self, "AI Source Required", "Set Prompt Source to Grok API or OpenAI API.")
            return
        if not concept:
            QMessageBox.warning(self, "Missing Concept", "Please enter a concept first.")
            return

        try:
            instruction = concept + " please turn this into a detailed 10 second prompt for grok imagine"
            system = "You are an expert prompt and social metadata generator for short-form AI videos. Return strict JSON only."
            user = (
                "Generate JSON with keys: manual_prompt, title, medium_title, tiktok_subheading, description, hashtags, category. "
                "manual_prompt should be detailed and cinematic for a 10-second Grok Imagine clip. "
                "title should be short and catchy. description should be 1-3 sentences. "
                "medium_title should be a medium-length title fit for social display. "
                "tiktok_subheading should be a slogan/subheading near 120 characters (roughly 100-140). "
                "hashtags should be an array of 5-12 hashtag strings without # prefixes. "
                "category should be the best YouTube category id as a string (default 22 if unsure). "
                f"Concept instruction: {instruction}"
            )
            raw = self._call_selected_ai(system, user)
            parsed = _parse_json_object_from_text(raw)
            manual_prompt = str(parsed.get("manual_prompt", "")).strip()
            if not manual_prompt:
                raise RuntimeError("AI response did not include a manual_prompt.")

            hashtags = parsed.get("hashtags", [])
            cleaned_hashtags = [str(tag).strip().lstrip("#") for tag in hashtags if str(tag).strip()]
            self.ai_social_metadata = AISocialMetadata(
                title=str(parsed.get("title", "AI Generated Video")).strip() or "AI Generated Video",
                medium_title=str(parsed.get("medium_title", parsed.get("title", "AI Generated Video Clip"))).strip() or "AI Generated Video Clip",
                tiktok_subheading=str(parsed.get("tiktok_subheading", "Swipe for more AI visuals.")).strip() or "Swipe for more AI visuals.",
                description=str(parsed.get("description", "")).strip(),
                hashtags=cleaned_hashtags or ["grok", "ai", "generated-video"],
                category=str(parsed.get("category", "22")).strip() or "22",
            )
            self.manual_prompt.setPlainText(manual_prompt)
            self._append_log(
                "AI updated Manual Prompt and social metadata defaults "
                f"(title/category/hashtags: {self.ai_social_metadata.title}/{self.ai_social_metadata.category}/"
                f"{', '.join(self.ai_social_metadata.hashtags)})."
            )
        except json.JSONDecodeError:
            QMessageBox.critical(self, "AI Response Error", "AI response was not valid JSON. Please retry.")
        except Exception as exc:
            QMessageBox.critical(self, "Prompt Generation Failed", str(exc))

    def populate_video_prompt(self) -> None:
        self.stop_all_requested = False
        manual_prompt = self.manual_prompt.toPlainText().strip()
        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return
        self._start_manual_browser_generation(manual_prompt, self.count.value())

    def start_image_generation(self) -> None:
        self.stop_all_requested = False
        manual_prompt = self.manual_prompt.toPlainText().strip()

        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return
        
        self._start_manual_browser_image_generation(manual_prompt, self.count.value())

    def start_sora_video_generation(self) -> None:
        self.stop_all_requested = False
        manual_prompt = self.manual_prompt.toPlainText().strip()

        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Please enter a manual prompt.")
            return

        if hasattr(self, "sora_browser_tab_index"):
            self.browser_tabs.setCurrentIndex(self.sora_browser_tab_index)
            self.browser = self.sora_browser

        self._append_log("Starting Sora video prompt run (video-only mode, no image-mode toggles).")
        self._start_manual_browser_generation(manual_prompt, self.count.value())

    def _start_manual_browser_generation(self, prompt: str, count: int) -> None:
        self.manual_generation_queue = [{"variant": idx} for idx in range(1, count + 1)]
        self._start_automation_counter_tracking(count)
        self._append_log(
            "Manual mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        self._append_log(f"Manual mode queued with counter={count}.")
        self._append_log("Attempting to populate the visible Grok prompt box on the current page...")
        self._submit_next_manual_variant()

    def _active_manual_browser_target(self) -> str:
        return "sora" if self.browser is self.sora_browser else "grok"

    def _return_embedded_browser_after_download(self) -> None:
        redirect_target = (self.pending_manual_redirect_target or "grok").lower()
        if redirect_target == "sora":
            self._append_log("Download complete; returning embedded browser to sora.chatgpt.com/drafts.")
            QTimer.singleShot(0, self.show_sora_browser_page)
            return

        self._append_log("Download complete; returning embedded browser to grok.com/imagine.")
        QTimer.singleShot(0, self.show_browser_page)

    def _start_manual_browser_image_generation(self, prompt: str, count: int) -> None:
        self.manual_image_generation_queue = [{"variant": idx} for idx in range(1, count + 1)]
        self._start_automation_counter_tracking(count)
        self._append_log(
            "Manual image mode now reuses the current browser page exactly as-is. "
            "No navigation or reload will happen."
        )
        self._append_log(f"Manual image mode queued with counter={count}.")
        self._submit_next_manual_image_variant()

    def _submit_next_manual_image_variant(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        if not self.manual_image_generation_queue:
            self._append_log("Manual browser image generation complete.")
            self._reset_automation_counter_tracking()
            return

        item = self.manual_image_generation_queue.pop(0)
        variant = item["variant"]
        attempts = int(item.get("attempts", 0)) + 1
        item["attempts"] = attempts
        prompt = self.manual_prompt.toPlainText().strip()
        if not prompt:
            self._append_log(
                f"ERROR: Manual image variant {variant} skipped because the Manual Prompt box is empty."
            )
            QTimer.singleShot(0, self._submit_next_manual_image_variant)
            return
        self.pending_manual_variant_for_download = variant
        self.pending_manual_download_type = "image"
        self.pending_manual_image_prompt = prompt
        self.pending_manual_redirect_target = self._active_manual_browser_target()
        self.manual_image_pick_clicked = False
        self.manual_image_video_mode_selected = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_video_mode_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token += 1
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")
        selected_duration_seconds = int(self.video_duration.currentData() or 10)
        selected_resolution_value = str(self.video_resolution.currentData() or "1280x720")
        selected_resolution_name = {"854x480": "480p", "1280x720": "720p"}.get(selected_resolution_value, "720p")

        populate_script = rf"""
            (() => {{
                try {{
                    const prompt = {prompt!r};
                    const selectors = [
                        "textarea[placeholder*='Type to imagine' i]",
                        "input[placeholder*='Type to imagine' i]",
                        "textarea[placeholder*='Type to customize this video' i]",
                        "input[placeholder*='Type to customize this video' i]",
                        "textarea[placeholder*='Type to customize video' i]",
                        "input[placeholder*='Type to customize video' i]",
                        "textarea[placeholder*='Customize video' i]",
                        "input[placeholder*='Customize video' i]",
                        "div.tiptap.ProseMirror[contenteditable='true']",
                        "[contenteditable='true'][aria-label*='Type to imagine' i]",
                        "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                        "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                        "[contenteditable='true'][aria-label*='Type to customize video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                        "[contenteditable='true'][aria-label*='Make a video' i]",
                        "[contenteditable='true'][data-placeholder*='Customize video' i]"
                    ];
                    const promptInput = selectors.map((sel) => document.querySelector(sel)).find(Boolean);
                    if (!promptInput) return {{ ok: false, error: "Prompt input not found" }};

                    promptInput.focus();
                    if (promptInput.isContentEditable) {{
                        const paragraph = document.createElement("p");
                        paragraph.textContent = prompt;
                        promptInput.replaceChildren(paragraph);
                        promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }} else {{
                        const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(promptInput), "value")?.set;
                        if (setter) setter.call(promptInput, prompt);
                        else promptInput.value = prompt;
                        promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }}
                    const typedValue = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                    if (!typedValue.trim()) return {{ ok: false, error: "Prompt field did not accept text" }};
                    return {{ ok: true, filledLength: typedValue.length }};
                }} catch (err) {{
                    return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                }}
            }})()
        """

        set_image_mode_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };
                    const textOf = (el) => (el?.textContent || "").replace(/\\s+/g, " ").trim();
                    const hasImageSelectionMarker = () => {
                        const selectedEls = [...document.querySelectorAll("[aria-selected='true'], [aria-pressed='true'], [data-state='checked'], [data-selected='true']")]
                            .filter((el) => isVisible(el));
                        return selectedEls.some((el) => /(^|\s)image(\s|$)/i.test(textOf(el)));
                    };

                    const modelTriggerCandidates = [
                        ...document.querySelectorAll("#model-select-trigger"),
                        ...document.querySelectorAll("button[aria-haspopup='menu'], [role='button'][aria-haspopup='menu']"),
                        ...document.querySelectorAll("button, [role='button']"),
                    ].filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el) && !looksLikeEditImageControl(el));

                    const modelTrigger = modelTriggerCandidates.find((el) => {
                        const txt = textOf(el);
                        return /model|video|image|options|settings/i.test(txt) || (el.id || "") === "model-select-trigger";
                    }) || null;

                    let optionsOpened = false;
                    if (modelTrigger) {
                        optionsOpened = emulateClick(modelTrigger);
                    }

                    const menuItemSelectors = [
                        "[role='menuitem'][data-radix-collection-item]",
                        "[role='menuitemradio']",
                        "[role='menuitem']",
                        "[role='option']",
                        "[data-radix-collection-item]",
                    ];

                    const menuItems = menuItemSelectors
                        .flatMap((sel) => [...document.querySelectorAll(sel)])
                        .filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el));

                    const imageItem = menuItems.find((el) => {
                        const txt = textOf(el);
                        return /(^|\s)image(\s|$)/i.test(txt) || /generate multiple images/i.test(txt);
                    }) || null;

                    const imageClicked = imageItem ? emulateClick(imageItem) : false;

                    const triggerNowSaysImage = !!(modelTrigger && /(^|\s)image(\s|$)/i.test(textOf(modelTrigger)));
                    const imageSelected = imageClicked || hasImageSelectionMarker() || triggerNowSaysImage;

                    return {
                        ok: true,
                        imageSelected,
                        optionsOpened,
                        imageItemFound: !!imageItem,
                        imageClicked,
                        triggerText: modelTrigger ? textOf(modelTrigger) : "",
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        set_image_options_script = r"""
            (() => {
                try {
                    const desiredAspect = "{selected_aspect_ratio}";
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const interactiveSelector = "button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label, span, div";
                    const textOf = (el) => (el?.textContent || "").replace(/\\s+/g, " ").trim();
                    const clickableAncestor = (el) => {
                        if (!el) return null;
                        if (typeof el.closest === "function") {
                            const ancestor = el.closest("button, [role='button'], [role='tab'], [role='option'], [role='menuitemradio'], [role='radio'], label");
                            if (ancestor) return ancestor;
                        }
                        return el;
                    };
                    const visibleTextElements = (root = document) => [...root.querySelectorAll(interactiveSelector)]
                        .filter((el) => isVisible(el) && textOf(el));
                    const selectedTextElements = (root = document) => visibleTextElements(root)
                        .filter((el) => {
                            const target = clickableAncestor(el);
                            if (!target) return false;
                            const ariaPressed = target.getAttribute("aria-pressed") === "true";
                            const ariaSelected = target.getAttribute("aria-selected") === "true";
                            const dataState = (target.getAttribute("data-state") || "").toLowerCase() === "checked";
                            const dataSelected = target.getAttribute("data-selected") === "true";
                            const classSelected = /\b(active|selected|checked|on)\b/i.test(target.className || "");
                            const checkedInput = !!target.querySelector("input[type='radio']:checked, input[type='checkbox']:checked");
                            return ariaPressed || ariaSelected || dataState || dataSelected || checkedInput || classSelected;
                        });

                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        const common = { bubbles: true, cancelable: true, composed: true };
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const matchesAny = (text, patterns) => patterns.some((pattern) => pattern.test(text));
                    const clickByText = (patterns, root = document) => {
                        const candidate = visibleTextElements(root).find((el) => matchesAny(textOf(el), patterns));
                        const target = clickableAncestor(candidate);
                        if (!target) return false;
                        return emulateClick(target);
                    };
                    const hasSelectedByText = (patterns, root = document) => selectedTextElements(root)
                        .some((el) => matchesAny(textOf(el), patterns));

                    const promptInput = document.querySelector("textarea[placeholder*='Describe your video' i], textarea[aria-label*='Describe your video' i], textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i]");
                    const composer = (promptInput && (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section"))) || document;

                    const aspectPatterns = {
                        "2:3": [/^2\s*:\s*3$/i],
                        "3:2": [/^3\s*:\s*2$/i],
                        "1:1": [/^1\s*:\s*1$/i],
                        "9:16": [/^9\s*:\s*16$/i],
                        "16:9": [/^16\s*:\s*9$/i],
                    };
                    const imagePatterns = [/(^|\s)image(\s|$)/i, /generate multiple images/i];
                    const desiredAspectPatterns = aspectPatterns[desiredAspect] || aspectPatterns["16:9"];

                    const optionsRequested = [];
                    const optionsApplied = [];

                    const findVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const candidates = [...root.querySelectorAll(`button[aria-label='${ariaLabel}']`)];
                        return candidates.find((el) => isVisible(el) && !el.disabled) || null;
                    };
                    const isOptionButtonSelected = (button) => {
                        if (!button) return false;
                        const ariaPressed = button.getAttribute("aria-pressed") === "true";
                        const ariaSelected = button.getAttribute("aria-selected") === "true";
                        const dataSelected = button.getAttribute("data-selected") === "true";
                        const dataState = (button.getAttribute("data-state") || "").toLowerCase();
                        if (ariaPressed || ariaSelected || dataSelected || dataState === "checked" || dataState === "active") return true;
                        if (/\b(active|selected|checked|on|text-fg-primary)\b/i.test(button.className || "")) return true;
                        const selectedFill = button.querySelector(".bg-primary:not([class*='bg-primary/'])");
                        return !!selectedFill;
                    };
                    const hasSelectedByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root);
                        return isOptionButtonSelected(button);
                    };
                    const clickVisibleButtonByAriaLabel = (ariaLabel, root = document) => {
                        const button = findVisibleButtonByAriaLabel(ariaLabel, root) || findVisibleButtonByAriaLabel(ariaLabel, document);
                        if (!button) return false;
                        return emulateClick(button);
                    };

                    const applyOption = (name, patterns, ariaLabel = null) => {
                        const alreadySelected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (alreadySelected) {
                            optionsApplied.push(`${name}(already-selected)`);
                            return;
                        }
                        const clicked = (ariaLabel && (clickVisibleButtonByAriaLabel(ariaLabel, composer) || clickVisibleButtonByAriaLabel(ariaLabel)))
                            || clickByText(patterns, composer)
                            || clickByText(patterns);
                        if (clicked) optionsRequested.push(name);
                        const selected = (ariaLabel && (hasSelectedByAriaLabel(ariaLabel, composer) || hasSelectedByAriaLabel(ariaLabel)))
                            || hasSelectedByText(patterns, composer)
                            || hasSelectedByText(patterns);
                        if (selected) optionsApplied.push(name);
                    };

                    applyOption("image", imagePatterns);
                    applyOption(desiredAspect, desiredAspectPatterns, desiredAspect);

                    const missingOptions = [];
                    if (!(hasSelectedByText(imagePatterns, composer) || hasSelectedByText(imagePatterns))) {
                        missingOptions.push("image");
                    }
                    if (!(hasSelectedByAriaLabel(desiredAspect, composer) || hasSelectedByAriaLabel(desiredAspect)
                        || hasSelectedByText(desiredAspectPatterns, composer)
                        || hasSelectedByText(desiredAspectPatterns))) {
                        missingOptions.push(desiredAspect);
                    }

                    return {
                        ok: true,
                        desiredAspect,
                        optionsRequested,
                        optionsApplied,
                        missingOptions,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """
        set_image_options_script = set_image_options_script.replace('"{selected_aspect_ratio}"', json.dumps(selected_aspect_ratio))

        submit_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const promptInput = document.querySelector("textarea[placeholder*='Describe your video' i], textarea[aria-label*='Describe your video' i], textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i]");
                    const submitSelectors = [
                        "button[type='submit'][aria-label='Submit']",
                        "button[aria-label='Submit'][type='submit']",
                        "button[type='submit']",
                        "button[aria-label*='submit' i]",
                        "[role='button'][aria-label*='submit' i]"
                    ];

                    const candidates = [];
                    const collect = (root) => {
                        if (!root || typeof root.querySelectorAll !== "function") return;
                        submitSelectors.forEach((selector) => {
                            const matches = root.querySelectorAll(selector);
                            for (let i = 0; i < matches.length; i += 1) candidates.push(matches[i]);
                        });
                    };

                    const composerRoot = (promptInput && typeof promptInput.closest === "function")
                        ? (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section") || promptInput.parentElement)
                        : null;
                    collect(composerRoot);
                    collect(document);

                    const submitButton = [...new Set(candidates)].find((el) => isVisible(el));
                    if (!submitButton) return { ok: false, error: "Submit button not found" };
                    if (submitButton.disabled) {
                        return { ok: false, waiting: true, status: "submit-disabled" };
                    }

                    const clicked = emulateClick(submitButton);
                    return {
                        ok: clicked,
                        waiting: !clicked,
                        status: clicked ? "submit-clicked" : "submit-click-failed",
                        ariaLabel: submitButton.getAttribute("aria-label") || "",
                        disabled: !!submitButton.disabled,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        post_create_image_script = r"""
            (async () => {
                try {
                    const basePrompt = __PROMPT_JSON__;
                    const modelName = "grok-3";
                    const aspectRatio = "__ASPECT__";
                    const videoLength = __SECONDS__;
                    const resolutionName = "__RESOLUTION__";
                    const payload = {
                        temporary: true,
                        modelName,
                        message: (/--mode=/i.test(String(basePrompt || "").trim()) ? String(basePrompt || "").trim() : `${String(basePrompt || "").trim()} --mode=custom`),
                        enableSideBySide: true,
                        responseMetadata: {
                            experiments: [],
                            modelConfigOverride: {
                                modelMap: {
                                    videoGenModelConfig: {
                                        aspectRatio,
                                        videoLength,
                                        resolutionName,
                                    },
                                },
                            },
                        },
                    };

                    const response = await fetch("https://grok.com/rest/app-chat/conversations/new", {
                        method: "POST",
                        credentials: "include",
                        headers: {
                            "content-type": "application/json",
                            "accept": "application/json, text/plain, */*",
                        },
                        body: JSON.stringify(payload),
                    });

                    const responseText = await response.text();
                    return {
                        ok: response.ok,
                        status: response.status,
                        payloadSummary: { aspectRatio, videoLength, resolutionName },
                        responseSnippet: responseText.slice(0, 400),
                        error: response.ok ? "" : `HTTP ${response.status}`,
                    };
                } catch (err) {
                    return {
                        ok: false,
                        status: 0,
                        responseSnippet: "",
                        error: String(err && err.stack ? err.stack : err),
                    };
                }
            })()
        """
        post_create_image_script = post_create_image_script.replace("__PROMPT_JSON__", json.dumps(prompt))
        post_create_image_script = post_create_image_script.replace("__ASPECT__", json.dumps(selected_aspect_ratio))
        post_create_image_script = post_create_image_script.replace("__SECONDS__", str(selected_duration_seconds))
        post_create_image_script = post_create_image_script.replace("__RESOLUTION__", json.dumps(selected_resolution_name))
        post_image_payload_preview = {
            "temporary": True,
            "modelName": "grok-3",
            "message": prompt if "--mode=" in prompt else f"{prompt} --mode=custom",
            "enableSideBySide": True,
            "responseMetadata": {
                "modelConfigOverride": {
                    "modelMap": {
                        "videoGenModelConfig": {
                            "aspectRatio": selected_aspect_ratio,
                            "videoLength": selected_duration_seconds,
                            "resolutionName": selected_resolution_name,
                        }
                    }
                }
            },
        }

        def _retry_variant(reason: str) -> None:
            self._append_log(f"WARNING: Manual image variant {variant} attempt {attempts} failed: {reason}")
            if attempts >= 4:
                self._append_log(
                    f"ERROR: Could not prepare manual image variant {variant} after {attempts} attempts; skipping variant."
                )
                self._submit_next_manual_image_variant()
                return
            self.manual_image_generation_queue.insert(0, item)
            QTimer.singleShot(1200, self._submit_next_manual_image_variant)

        submit_attempts = 0

        def _run_submit_attempt() -> None:
            nonlocal submit_attempts
            submit_attempts += 1
            self._append_log(
                f"Manual image variant {variant}: attempting direct POST submit to https://grok.com/rest/app-chat/conversations/new "
                f"({submit_attempts}/12) with payload {json.dumps(post_image_payload_preview, ensure_ascii=False)}"
            )
            self.browser.page().runJavaScript(post_create_image_script, _after_direct_post_submit)

        def _after_direct_post_submit(result):
            post_status = result.get("status") if isinstance(result, dict) else "unknown"
            response_snippet = result.get("responseSnippet") if isinstance(result, dict) else ""
            self._append_log(
                f"Manual image variant {variant}: /conversations/new response status={post_status}; body snippet={response_snippet!r}"
            )

            if isinstance(result, dict) and result.get("ok"):
                payload_summary = result.get("payloadSummary") if isinstance(result, dict) else {}
                self._append_log(
                    f"Submitted manual image variant {variant} via direct POST "
                    f"(aspect={payload_summary.get('aspectRatio') if isinstance(payload_summary, dict) else 'unknown'}, "
                    f"duration={payload_summary.get('videoLength') if isinstance(payload_summary, dict) else 'unknown'}s, "
                    f"resolution={payload_summary.get('resolutionName') if isinstance(payload_summary, dict) else 'unknown'}). "
                    "Waiting 2.0s before continuing to image polling."
                )
                QTimer.singleShot(2000, lambda: QTimer.singleShot(7000, self._poll_for_manual_image))
                return

            post_error = result.get("error") if isinstance(result, dict) else result
            self._append_log(
                f"WARNING: Manual image variant {variant}: direct POST submit failed: {post_error!r}; "
                "waiting 2.0s before fallback submit click."
            )
            QTimer.singleShot(2000, lambda: self.browser.page().runJavaScript(submit_script, _after_submit))

        def _after_submit(result):
            if isinstance(result, dict) and result.get("ok"):
                #self.show_browser_page()
                self._append_log(
                    f"Submitted manual image variant {variant} (attempt {attempts}); "
                    "waiting for first rendered image, then opening it for download."
                )
                QTimer.singleShot(7000, self._poll_for_manual_image)
                return

            if isinstance(result, dict) and result.get("waiting"):
                if submit_attempts < 12:
                    self._append_log(
                        f"Manual image variant {variant}: submit button still disabled (attempt {submit_attempts}); retrying click..."
                    )
                    QTimer.singleShot(500, _run_submit_attempt)
                    return
                _retry_variant(f"submit button stayed disabled: {result!r}")
                return

            # Some Grok navigations can clear the JS callback value; treat that as submitted.
            if result in (None, ""):
                #self.show_browser_page()
                self._append_log(
                    f"Submitted manual image variant {variant} (attempt {attempts}); "
                    "submit callback returned empty result after page activity; continuing to image polling."
                )
                QTimer.singleShot(7000, self._poll_for_manual_image)
                return

            _retry_variant(f"submit failed: {result!r}")

        def _after_set_mode(result):
            if result in (None, ""):
                self._append_log(
                    f"Manual image variant {variant}: image-mode callback returned empty result; "
                    "continuing with prompt population and assuming current mode is correct."
                )
            elif not isinstance(result, dict) or not result.get("ok"):
                _retry_variant(f"set image mode script failed: {result!r}")
                return
            elif not result.get("imageSelected"):
                _retry_variant(f"image option not selected: {result!r}")
                return

            self._append_log(
                "Manual image variant "
                f"{variant}: image mode selected={result.get('imageSelected') if isinstance(result, dict) else 'unknown'} "
                f"(opened={result.get('optionsOpened') if isinstance(result, dict) else 'unknown'}, "
                f"itemFound={result.get('imageItemFound') if isinstance(result, dict) else 'unknown'}, "
                f"itemClicked={result.get('imageClicked') if isinstance(result, dict) else 'unknown'}); "
                f"applying aspect option {selected_aspect_ratio} next (attempt {attempts})."
            )

            def _after_image_options(options_result):
                if not isinstance(options_result, dict) or not options_result.get("ok"):
                    self._append_log(
                        f"WARNING: Manual image variant {variant}: image options script failed; continuing. result={options_result!r}"
                    )
                else:
                    requested_summary = ", ".join(options_result.get("optionsRequested") or []) or "none"
                    applied_summary = ", ".join(options_result.get("optionsApplied") or []) or "none detected"
                    missing_summary = ", ".join(options_result.get("missingOptions") or []) or "none"
                    self._append_log(
                        f"Manual image variant {variant}: image options requested: {requested_summary}; "
                        f"applied markers: {applied_summary}; missing: {missing_summary}."
                    )

                QTimer.singleShot(450, lambda: self.browser.page().runJavaScript(populate_script, _after_populate))

            QTimer.singleShot(450, lambda: self.browser.page().runJavaScript(set_image_options_script, _after_image_options))

        submit_delay_ms = 2000

        def _after_populate(result):
            if result in (None, ""):
                self._append_log(
                    f"Manual image variant {variant}: prompt populate callback returned empty result; "
                    f"waiting {submit_delay_ms / 1000:.1f}s before submit."
                )
                QTimer.singleShot(submit_delay_ms, _run_submit_attempt)
                return

            if not isinstance(result, dict) or not result.get("ok"):
                _retry_variant(f"prompt population failed: {result!r}")
                return

            self._append_log(
                f"Manual image variant {variant}: prompt populated (length={result.get('filledLength', 'unknown')}); "
                f"waiting {submit_delay_ms / 1000:.1f}s before submitting prompt."
            )
            QTimer.singleShot(submit_delay_ms, _run_submit_attempt)

        self.browser.page().runJavaScript(set_image_mode_script, _after_set_mode)

    def _poll_for_manual_image(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        variant = self.pending_manual_variant_for_download
        if variant is None or self.pending_manual_download_type != "image":
            return

        prompt = self.pending_manual_image_prompt or ""
        if not self.manual_image_pick_clicked:
            phase = "pick"
        elif not self.manual_image_video_mode_selected:
            phase = "video-mode"
        else:
            phase = "submit"
        script = f"""
            (async () => {{
                const prompt = {prompt!r};
                const phase = {phase!r};
                const submitToken = {self.manual_image_submit_token};
                const ACTION_DELAY_MS = 200;
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const common = {{ bubbles: true, cancelable: true, composed: true }};
                const emulateClick = (el) => {{
                    if (!el || !isVisible(el) || el.disabled) return false;
                    try {{ el.dispatchEvent(new PointerEvent("pointerdown", common)); }} catch (_) {{}}
                    el.dispatchEvent(new MouseEvent("mousedown", common));
                    try {{ el.dispatchEvent(new PointerEvent("pointerup", common)); }} catch (_) {{}}
                    el.dispatchEvent(new MouseEvent("mouseup", common));
                    el.dispatchEvent(new MouseEvent("click", common));
                    return true;
                }};

                if (phase === "pick") {{
                    const listItemOf = (el) => el?.closest("[role='listitem'], li, article, figure") || null;

                    const customizePromptReady = !!document.querySelector(
                        "textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], "
                        + "[contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i]"
                    );
                    if (customizePromptReady) {{
                        return {{ ok: true, status: "generated-image-clicked" }};
                    }}

                    const makeVideoButtons = [...document.querySelectorAll("[role='listitem'] button[aria-label*='make video' i]")]
                        .filter((btn) => isVisible(btn) && !btn.disabled);

                    if (makeVideoButtons.length) {{
                        makeVideoButtons.sort((a, b) => {{
                            const ar = a.getBoundingClientRect();
                            const br = b.getBoundingClientRect();
                            const rowDelta = Math.abs(ar.top - br.top);
                            if (rowDelta > 20) return ar.top - br.top;
                            return ar.left - br.left;
                        }});

                        const firstButton = makeVideoButtons[0];
                        const tile = listItemOf(firstButton) || firstButton.parentElement;
                        const tileImage = tile?.querySelector?.("img") || null;
                        const clickedTile = emulateClick(tileImage) || emulateClick(tile) || emulateClick(firstButton);
                        if (!clickedTile) return {{ ok: false, status: "generated-image-click-failed" }};
                        await sleep(ACTION_DELAY_MS);
                        return {{ ok: true, status: "generated-image-clicked" }};
                    }}

                    const listItemImages = [...document.querySelectorAll("[role='listitem'] img")]
                        .filter((img) => {{
                            if (!isVisible(img)) return false;
                            const alt = (img.getAttribute("alt") || "").trim().toLowerCase();
                            const title = (img.getAttribute("title") || "").trim().toLowerCase();
                            const aria = (img.getAttribute("aria-label") || "").trim().toLowerCase();
                            const descriptor = (alt + " " + title + " " + aria).trim();
                            if (/\bedit\\s+image\b/i.test(descriptor)) return false;
                            const width = img.naturalWidth || img.width || img.clientWidth || 0;
                            const height = img.naturalHeight || img.height || img.clientHeight || 0;
                            return width >= 220 && height >= 220;
                        }});

                    if (!listItemImages.length) return {{ ok: false, status: "waiting-for-generated-image" }};

                    listItemImages.sort((a, b) => {{
                        const ar = a.getBoundingClientRect();
                        const br = b.getBoundingClientRect();
                        const rowDelta = Math.abs(ar.top - br.top);
                        if (rowDelta > 20) return ar.top - br.top;
                        return ar.left - br.left;
                    }});

                    const firstImage = listItemImages[0];
                    const listItem = listItemOf(firstImage) || firstImage.closest("button, [role='button']") || firstImage.parentElement;
                    const clickedImage = emulateClick(firstImage) || emulateClick(listItem);
                    if (!clickedImage) return {{ ok: false, status: "generated-image-click-failed" }};
                    await sleep(ACTION_DELAY_MS);
                    return {{ ok: true, status: "generated-image-clicked" }};
                }}

                if (phase === "video-mode") {{
                    const textOf = (el) => (el?.textContent || "").replace(/\\s+/g, " ").trim();
                    const ariaOf = (el) => (el?.getAttribute?.("aria-label") || "").replace(/\\s+/g, " ").trim();
                    const looksLikeEditImageControl = (el) => /\\bedit\\s+image\\b/i.test(`${{textOf(el)}} ${{ariaOf(el)}}`);
                    const modelTriggerCandidates = [
                        ...document.querySelectorAll("#model-select-trigger"),
                        ...document.querySelectorAll("button[aria-haspopup='menu'], [role='button'][aria-haspopup='menu']"),
                        ...document.querySelectorAll("button, [role='button']"),
                    ].filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el) && !looksLikeEditImageControl(el));

                    const modelTrigger = modelTriggerCandidates.find((el) => {{
                        const txt = textOf(el);
                        return /model|video|image|options|settings/i.test(txt) || (el.id || "") === "model-select-trigger";
                    }}) || null;

                    let optionsOpened = false;
                    if (modelTrigger) {{
                        optionsOpened = emulateClick(modelTrigger);
                    }}
                    await sleep(ACTION_DELAY_MS);

                    const menuItems = [
                        ...document.querySelectorAll("[role='menuitem'][data-radix-collection-item], [role='menuitemradio'], [role='menuitem'], [role='option'], [data-radix-collection-item]")
                    ].filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el));

                    const videoItem = menuItems.find((el) => /(^|\\s)video(\\s|$)/i.test(textOf(el))) || null;
                    const videoClicked = videoItem ? emulateClick(videoItem) : false;
                    await sleep(ACTION_DELAY_MS);

                    const selectedEls = [...document.querySelectorAll("[aria-selected='true'], [aria-pressed='true'], [data-state='checked'], [data-selected='true']")]
                        .filter((el) => isVisible(el));
                    const selectedViaMarker = selectedEls.some((el) => /(^|\\s)video(\\s|$)/i.test(textOf(el)));
                    const selectedViaTrigger = !!(modelTrigger && /(^|\\s)video(\\s|$)/i.test(textOf(modelTrigger)));
                    const videoSelected = videoClicked || selectedViaMarker || selectedViaTrigger;

                    if (!videoSelected) {{
                        return {{
                            ok: false,
                            status: "waiting-for-video-mode",
                            optionsOpened,
                            videoItemFound: !!videoItem,
                            videoClicked,
                        }};
                    }}

                    return {{
                        ok: true,
                        status: "video-mode-selected",
                        optionsOpened,
                        videoItemFound: !!videoItem,
                        videoClicked,
                    }};
                }}

                if (window.__grokManualImageSubmitToken === submitToken) {{
                    return {{ ok: true, status: "video-submit-already-clicked" }};
                }}

                const promptSelectors = [
                    "textarea[placeholder*='Type to customize video' i]",
                    "input[placeholder*='Type to customize video' i]",
                    "textarea[placeholder*='Type to imagine' i]",
                    "input[placeholder*='Type to imagine' i]",
                    "div.tiptap.ProseMirror[contenteditable='true']",
                    "[contenteditable='true'][aria-label*='Type to customize video' i]",
                    "[contenteditable='true'][aria-label*='Type to imagine' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                ];
                const promptInput = promptSelectors.map((sel) => document.querySelector(sel)).find(Boolean);
                if (!promptInput) return {{ ok: false, status: "image-clicked-waiting-prompt-input" }};

                promptInput.focus();
                await sleep(ACTION_DELAY_MS);

                if (promptInput.isContentEditable) {{
                    const paragraph = document.createElement("p");
                    paragraph.textContent = prompt;
                    promptInput.replaceChildren(paragraph);
                }} else {{
                    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(promptInput), "value")?.set;
                    if (setter) setter.call(promptInput, prompt);
                    else promptInput.value = prompt;
                }}
                await sleep(ACTION_DELAY_MS);

                promptInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
                await sleep(ACTION_DELAY_MS);
                promptInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
                await sleep(ACTION_DELAY_MS);

                const typedValue = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                if (!typedValue.trim()) return {{ ok: false, status: "prompt-fill-empty" }};

                const submitButton = [...document.querySelectorAll("button[type='submit'], button[aria-label*='submit' i], button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /submit|make\\s+video/i.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));
                if (!submitButton) return {{ ok: false, status: "prompt-filled-waiting-submit" }};

                await sleep(ACTION_DELAY_MS);
                const submitted = emulateClick(submitButton);
                if (submitted) window.__grokManualImageSubmitToken = submitToken;
                return {{
                    ok: submitted,
                    status: submitted ? "video-submit-clicked" : "submit-click-failed",
                    buttonLabel: (submitButton.getAttribute("aria-label") || submitButton.textContent || "").trim(),
                    filledLength: typedValue.length,
                }};
            }})()
        """


        def _after_poll(result):
            current_variant = self.pending_manual_variant_for_download
            if current_variant is None:
                return

            if isinstance(result, dict) and result.get("ok"):
                status = result.get("status") or "ok"
                if status == "generated-image-clicked":
                    if not self.manual_image_pick_clicked:
                        self._append_log(
                            f"Variant {current_variant}: clicked first generated image tile; preparing video prompt + submit."
                        )
                    self.manual_image_pick_clicked = True
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_video_mode_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    QTimer.singleShot(1000, self._poll_for_manual_image)
                    return

                if status == "video-mode-selected":
                    opened = result.get("optionsOpened")
                    item_found = result.get("videoItemFound")
                    clicked = result.get("videoClicked")
                    if not self.manual_image_video_mode_selected:
                        self._append_log(
                            f"Variant {current_variant}: switched prompt mode to video "
                            f"(opened={opened}, itemFound={item_found}, itemClicked={clicked}); refilling prompt."
                        )
                    self.manual_image_video_mode_selected = True
                    self.manual_image_video_mode_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    QTimer.singleShot(700, self._poll_for_manual_image)
                    return

                if status in ("video-submit-clicked", "video-submit-already-clicked"):
                    if status == "video-submit-clicked":
                        detail = result.get("buttonLabel") or "submit"
                        filled_length = result.get("filledLength")
                        if isinstance(filled_length, int):
                            message = f"video prompt submitted via '{detail}' (length={filled_length})"
                        else:
                            message = f"video prompt submitted via '{detail}'"
                    else:
                        message = "submit was already clicked earlier; waiting for video render/download"

                    if not self.manual_image_video_submit_sent:
                        self._append_log(f"Variant {current_variant}: {message}.")
                    self.manual_image_video_submit_sent = True
                    self.manual_image_video_mode_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    self.pending_manual_download_type = "video"
                    self._trigger_browser_video_download(current_variant, allow_make_video_click=False)
                    return

            status = result.get("status") if isinstance(result, dict) else "callback-empty"
            if not self.manual_image_pick_clicked:
                def _queue_pick_retry(current_status: str) -> None:
                    self.manual_image_pick_retry_count += 1
                    if self.manual_image_pick_retry_count >= self.MANUAL_IMAGE_PICK_RETRY_LIMIT:
                        if current_status == "callback-empty":
                            self._append_log(
                                "WARNING: Variant "
                                f"{current_variant}: image pick stayed callback-empty for "
                                f"{self.manual_image_pick_retry_count} checks; assuming pick stage already completed and advancing."
                            )
                            self.manual_image_pick_clicked = True
                            self.manual_image_video_mode_selected = True
                            self.manual_image_pick_retry_count = 0
                            self.manual_image_video_mode_retry_count = 0
                            self.manual_image_submit_retry_count = 0
                            QTimer.singleShot(700, self._poll_for_manual_image)
                            return
                        self._append_log(
                            "WARNING: Variant "
                            f"{current_variant}: image pick validation stayed in '{current_status}' for "
                            f"{self.manual_image_pick_retry_count} checks; continuing to wait for pick state."
                        )
                        self.manual_image_pick_retry_count = 0
                    self._append_log(
                        f"Variant {current_variant}: generated image not ready for pick+submit yet ({current_status}); retrying..."
                    )
                    QTimer.singleShot(3000, self._poll_for_manual_image)

                if status == "callback-empty":
                    pick_ready_probe_script = """
                        (() => {
                            try {
                                const customizePromptVisible = !!document.querySelector(
                                    "textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], " +
                                    "[contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i]"
                                );
                                const makeVideoButtonVisible = !![...document.querySelectorAll("button[aria-label*='make video' i]")]
                                    .find((btn) => !!(btn && (btn.offsetWidth || btn.offsetHeight || btn.getClientRects().length)));
                                const editImageVisible = !![...document.querySelectorAll("button[aria-label*='edit image' i], [role='button'][aria-label*='edit image' i]")]
                                    .find((el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length)));
                                const path = String((window.location && window.location.pathname) || "").toLowerCase();
                                const onPostView = path.includes("/imagine/post/");
                                const ready = customizePromptVisible || (onPostView && (editImageVisible || makeVideoButtonVisible));
                                return {
                                    ready,
                                    customizePromptVisible,
                                    makeVideoButtonVisible,
                                    editImageVisible,
                                    onPostView,
                                };
                            } catch (_) {
                                return { ready: false };
                            }
                        })()
                    """

                    def _after_pick_ready_probe(probe_result):
                        if isinstance(probe_result, dict) and probe_result.get("ready"):
                            ready_flags = (
                                f"customizePrompt={bool(probe_result.get('customizePromptVisible'))}, "
                                f"makeVideoBtn={bool(probe_result.get('makeVideoButtonVisible'))}, "
                                f"editImage={bool(probe_result.get('editImageVisible'))}, "
                                f"postView={bool(probe_result.get('onPostView'))}"
                            )
                            self._append_log(
                                f"Variant {current_variant}: detected post/customize UI after empty callback ({ready_flags}); treating image pick as complete."
                            )
                            self.manual_image_pick_clicked = True
                            self.manual_image_video_mode_selected = True
                            self.manual_image_pick_retry_count = 0
                            self.manual_image_video_mode_retry_count = 0
                            self.manual_image_submit_retry_count = 0
                            QTimer.singleShot(700, self._poll_for_manual_image)
                            return
                        _queue_pick_retry(status)

                    self.browser.page().runJavaScript(pick_ready_probe_script, _after_pick_ready_probe)
                    return

                _queue_pick_retry(status)
                return

            if not self.manual_image_video_mode_selected:
                self.manual_image_video_mode_retry_count += 1
                if self.manual_image_video_mode_retry_count >= self.MANUAL_IMAGE_SUBMIT_RETRY_LIMIT:
                    self._append_log(
                        "WARNING: Variant "
                        f"{current_variant}: video-mode validation stayed in '{status}' for "
                        f"{self.manual_image_video_mode_retry_count} checks; continuing to wait for video-mode state."
                    )
                    self.manual_image_video_mode_retry_count = 0

                self._append_log(
                    f"Variant {current_variant}: waiting for video mode selection ({status}); retrying..."
                )
                QTimer.singleShot(2500, self._poll_for_manual_image)
                return

            self.manual_image_submit_retry_count += 1
            if self.manual_image_submit_retry_count >= self.MANUAL_IMAGE_SUBMIT_RETRY_LIMIT:
                self._append_log(
                    "WARNING: Variant "
                    f"{current_variant}: submit-stage validation stayed in '{status}' for "
                    f"{self.manual_image_submit_retry_count} checks; assuming submit succeeded and continuing to download polling."
                )
                self.manual_image_video_submit_sent = True
                self.manual_image_submit_retry_count = 0
                self.pending_manual_download_type = "video"
                self._trigger_browser_video_download(current_variant, allow_make_video_click=False)
                return

            self._append_log(
                f"Variant {current_variant}: video submit stage not ready yet ({status}); retrying..."
            )
            QTimer.singleShot(3000, self._poll_for_manual_image)

        self.browser.page().runJavaScript(script, _after_poll)

    def _start_continue_iteration(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        frame_path: Path | None = None
        if self.continue_from_frame_seed_image_path is not None:
            frame_path = self.continue_from_frame_seed_image_path
            self._append_log(f"Continue-from-image: using selected image: {frame_path}")
        else:
            latest_video = self._resolve_latest_video_for_continuation()
            if not latest_video:
                self._append_log("ERROR: No videos available to continue from last frame.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return

            self.continue_from_frame_current_source_video = latest_video
            self._append_log(f"Continue-from-last-frame: extracting frame from source video: {latest_video}")
            frame_path = self._extract_last_frame(latest_video)
            if frame_path is None:
                self._append_log("ERROR: Continue-from-last-frame stopped because frame extraction failed.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return
            self._append_log(f"Continue-from-last-frame: extracted last frame to {frame_path}")
            if not self._copy_image_to_clipboard(frame_path):
                self._append_log("ERROR: Continue-from-last-frame stopped because clipboard image copy failed.")
                self.continue_from_frame_active = False
                self.continue_from_frame_current_source_video = ""
                return

        self.last_extracted_frame_path = frame_path
        iteration = self.continue_from_frame_completed + 1
        self._append_log(
            f"Continue iteration {iteration}/{self.continue_from_frame_target_count}: using seed image {frame_path}"
        )
        browser_page_pause_ms = 200
        self._append_log(
            "Continue mode: starting image paste into the current Grok prompt area without forcing page navigation..."
        )
        QTimer.singleShot(
            9000 + browser_page_pause_ms,
            lambda: self._upload_frame_into_grok(frame_path, on_uploaded=self._wait_for_continue_upload_reload),
        )
        self._append_log(
            "Continue mode: image paste scheduled; waiting for upload/reload before prompt submission."
        )

    def _resolve_latest_video_for_continuation(self) -> str | None:
        in_session_files: list[Path] = []
        for video in self.videos:
            video_path_raw = video.get("video_file_path")
            if not video_path_raw:
                continue
            video_path = Path(str(video_path_raw))
            if video_path.is_file():
                in_session_files.append(video_path)

        if in_session_files:
            latest_in_session = max(in_session_files, key=lambda path: path.stat().st_mtime)
            return str(latest_in_session)

        candidates: list[Path] = []
        for pattern in ("*.mp4", "*.mov", "*.webm"):
            candidates.extend(self.download_dir.glob(pattern))

        files = [path for path in candidates if path.is_file()]
        if not files:
            return None

        latest = max(files, key=lambda path: path.stat().st_mtime)
        self._append_log(
            "Continue-from-last-frame fallback: no in-session videos found, "
            f"using latest file from downloads folder: {latest}"
        )
        return str(latest)

    def _submit_next_manual_variant(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        if not self.manual_generation_queue:
            self._append_log("Manual browser generation complete.")
            self._reset_automation_counter_tracking()
            return

        item = self.manual_generation_queue.pop(0)
        remaining_count = len(self.manual_generation_queue)
        variant = item["variant"]
        prompt = self.manual_prompt.toPlainText().strip()
        if not prompt:
            self._append_log(
                f"ERROR: Manual variant {variant} skipped because the Manual Prompt box is empty."
            )
            QTimer.singleShot(0, self._submit_next_manual_variant)
            return
        self.pending_manual_variant_for_download = variant
        self.pending_manual_download_type = "video"
        self.pending_manual_redirect_target = self._active_manual_browser_target()
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        action_delay_ms = 1000
        selected_quality_label = self.video_resolution.currentText().split(" ", 1)[0]
        selected_duration_seconds = int(self.video_duration.currentData() or 10)
        selected_duration_label = f"{selected_duration_seconds}s"
        selected_aspect_ratio = str(self.video_aspect_ratio.currentData() or "16:9")
        selected_resolution_value = str(self.video_resolution.currentData() or "1280x720")
        resolution_name_map = {
            "854x480": "480p",
            "1280x720": "720p",
        }
        selected_resolution_name = resolution_name_map.get(selected_resolution_value, "720p")
        self._append_log(
            f"Populating prompt for manual variant {variant} in browser, setting video options "
            f"({selected_quality_label}, {selected_aspect_ratio}), then force submitting with {action_delay_ms}ms delays between each action. "
            f"Remaining repeats after this: {remaining_count}."
        )

        post_create_video_script = r"""
            (async () => {
                try {
                    const basePrompt = __PROMPT_JSON__;
                    const aspectRatio = "__ASPECT__";
                    const videoLength = __SECONDS__;
                    const resolutionName = "__RESOLUTION__";
                    const modelName = "grok-3";

                    const normalizePrompt = (value) => {
                        const txt = String(value || "").trim();
                        if (!txt) return txt;
                        return /--mode=/i.test(txt) ? txt : `${txt} --mode=custom`;
                    };

                    const extractParentPostId = () => {
                        const uuidRegex = /[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/ig;
                        const hrefSources = [
                            window.location.href,
                            ...Array.from(document.querySelectorAll("a[href], button[data-post-id], [data-parent-post-id]"))
                                .map((el) => el.getAttribute("href") || el.getAttribute("data-post-id") || el.getAttribute("data-parent-post-id") || ""),
                        ];
                        for (const source of hrefSources) {
                            const matches = String(source || "").match(uuidRegex);
                            if (matches && matches.length) return matches[matches.length - 1];
                        }
                        return "";
                    };

                    const payload = {
                        temporary: true,
                        modelName,
                        message: normalizePrompt(basePrompt),
                        toolOverrides: { videoGen: true },
                        enableSideBySide: true,
                    };

                    const parentPostId = extractParentPostId();
                    payload.responseMetadata = {
                        experiments: [],
                        modelConfigOverride: {
                            modelMap: {
                                videoGenModelConfig: {
                                    aspectRatio,
                                    videoLength,
                                    resolutionName,
                                },
                            },
                        },
                    };
                    if (parentPostId) {
                        payload.responseMetadata.modelConfigOverride.modelMap.videoGenModelConfig.parentPostId = parentPostId;
                    }

                    const response = await fetch("https://grok.com/rest/app-chat/conversations/new", {
                        method: "POST",
                        credentials: "include",
                        headers: {
                            "content-type": "application/json",
                            "accept": "application/json, text/plain, */*",
                        },
                        body: JSON.stringify(payload),
                    });

                    const responseText = await response.text();
                    let responseJson = null;
                    try {
                        responseJson = responseText ? JSON.parse(responseText) : null;
                    } catch (_) {
                        responseJson = null;
                    }

                    return {
                        ok: response.ok,
                        status: response.status,
                        usedDirectPost: true,
                        parentPostId,
                        payloadSummary: { aspectRatio, videoLength, resolutionName },
                        responseSnippet: responseText.slice(0, 400),
                        conversationId: responseJson && (responseJson.conversationId || responseJson.id || ""),
                        error: response.ok ? "" : `HTTP ${response.status}`,
                    };
                } catch (err) {
                    return {
                        ok: false,
                        usedDirectPost: true,
                        error: String(err && err.stack ? err.stack : err),
                    };
                }
            })()
        """
        post_create_video_script = post_create_video_script.replace("__PROMPT_JSON__", json.dumps(prompt))
        post_create_video_script = post_create_video_script.replace("__ASPECT__", json.dumps(selected_aspect_ratio))
        post_create_video_script = post_create_video_script.replace("__SECONDS__", str(selected_duration_seconds))
        post_create_video_script = post_create_video_script.replace("__RESOLUTION__", json.dumps(selected_resolution_name))

        post_payload_preview = {
            "temporary": True,
            "modelName": "grok-3",
            "message": prompt if "--mode=" in prompt else f"{prompt} --mode=custom",
            "toolOverrides": {"videoGen": True},
            "enableSideBySide": True,
            "responseMetadata": {
                "modelConfigOverride": {
                    "modelMap": {
                        "videoGenModelConfig": {
                            "aspectRatio": selected_aspect_ratio,
                            "videoLength": selected_duration_seconds,
                            "resolutionName": selected_resolution_name,
                        }
                    }
                }
            },
        }
        self._append_log(
            f"Variant {variant}: attempting direct POST to https://grok.com/rest/app-chat/conversations/new "
            f"with payload {json.dumps(post_payload_preview, ensure_ascii=False)}"
        )

        escaped_prompt = repr(prompt)
        script = rf"""
            (() => {{
                try {{
                    const prompt = {escaped_prompt};
                    const promptSelectors = [
                        "textarea[placeholder*='Describe your video' i]",
                        "textarea[aria-label*='Describe your video' i]",
                        "textarea[placeholder*='Type to imagine' i]",
                        "input[placeholder*='Type to imagine' i]",
                        "textarea[placeholder*='Type to customize this video' i]",
                        "input[placeholder*='Type to customize this video' i]",
                        "textarea[placeholder*='Type to customize video' i]",
                        "input[placeholder*='Type to customize video' i]",
                        "textarea[placeholder*='Customize video' i]",
                        "input[placeholder*='Customize video' i]",
                        "div.tiptap.ProseMirror[contenteditable='true']",
                        "[contenteditable='true'][aria-label*='Type to imagine' i]",
                        "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                        "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                        "[contenteditable='true'][aria-label*='Type to customize video' i]",
                        "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                        "[contenteditable='true'][aria-label*='Make a video' i]",
                        "[contenteditable='true'][data-placeholder*='Customize video' i]"
                    ];

                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const inputCandidates = [];
                    promptSelectors.forEach((selector) => {{
                        const matches = document.querySelectorAll(selector);
                        for (let i = 0; i < matches.length; i += 1) inputCandidates.push(matches[i]);
                    }});
                    const input = inputCandidates.find((el) => isVisible(el));
                    if (!input) return {{ ok: false, error: "Prompt input not found" }};

                    const common = {{ bubbles: true, cancelable: true, composed: true }};
                    try {{ input.dispatchEvent(new PointerEvent("pointerdown", common)); }} catch (_) {{}}
                    input.dispatchEvent(new MouseEvent("mousedown", common));
                    try {{ input.dispatchEvent(new PointerEvent("pointerup", common)); }} catch (_) {{}}
                    input.dispatchEvent(new MouseEvent("mouseup", common));
                    input.dispatchEvent(new MouseEvent("click", common));
                    input.focus();
                    if (input.isContentEditable) {{
                        // Only populate the field; do not synthesize Enter/submit key events.
                        const paragraph = document.createElement("p");
                        paragraph.textContent = prompt;
                        input.replaceChildren(paragraph);
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }} else {{
                        const proto = Object.getPrototypeOf(input);
                        const descriptor = proto ? Object.getOwnPropertyDescriptor(proto, "value") : null;
                        const valueSetter = descriptor && descriptor.set;
                        if (valueSetter) valueSetter.call(input, prompt);
                        else input.value = prompt;
                        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                        input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                    }}

                    const typedValue = input.isContentEditable ? (input.textContent || "") : (input.value || "");
                    if (!typedValue.trim()) return {{ ok: false, error: "Prompt field did not accept text" }};

                    return {{ ok: true, filledLength: typedValue.length }};
                }} catch (err) {{
                    return {{ ok: false, error: String(err && err.stack ? err.stack : err) }};
                }}
            }})()
        """

        open_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const isOptionsMenu = (el) => {
                        if (!el || !isVisible(el)) return false;
                        const txt = (el.textContent || "").trim();
                        return /video\s*duration|resolution|aspect\s*ratio|\borientation\b|\bduration\b|\bvideos\b|\bimage\b|\bvideo\b/i.test(txt);
                    };

                    const findOpenMenu = () => {
                        const candidates = [
                            ...document.querySelectorAll("[role='menu'][data-state='open']"),
                            ...document.querySelectorAll("[data-radix-menu-content][data-state='open']"),
                            ...document.querySelectorAll("[role='menu'], [data-radix-menu-content]")
                        ];
                        return candidates.find((el) => isOptionsMenu(el)) || null;
                    };

                    const settingsCandidates = [
                        ...document.querySelectorAll("button[aria-label='Settings'][id='radix-:rg:']"),
                        ...document.querySelectorAll("button[aria-haspopup='menu'][aria-label='Settings'][id='radix-:rg:']"),
                        ...document.querySelectorAll("button[id^='radix-'][aria-label='Settings'][id='radix-:rg:']")
                    ].filter((el, index, arr) => arr.indexOf(el) === index);
                    const settingsButton = settingsCandidates.find((el) => isVisible(el) && !el.disabled) || null;

                    if (!settingsButton) {
                        return { ok: false, error: "Settings button not found", panelVisible: !!findOpenMenu() };
                    }

                    const wasExpanded = settingsButton.getAttribute("aria-expanded") === "true";
                    const menuBefore = findOpenMenu();
                    const opened = menuBefore ? true : emulateClick(settingsButton);
                    const menu = findOpenMenu();
                    return {
                        ok: opened || !!menu,
                        opened,
                        panelVisible: !!menu,
                        triggerAriaLabel: settingsButton.getAttribute("aria-label") || "",
                        triggerId: settingsButton.id || "",
                        wasExpanded,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        verify_prompt_script = r"""
            (() => {
                try {
                    const promptInput = document.querySelector("textarea[placeholder*='Describe your video' i], textarea[aria-label*='Describe your video' i], textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], textarea[aria-label*='Make a video' i], input[aria-label*='Make a video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i], [contenteditable='true'][aria-label*='Type to customize this video' i], [contenteditable='true'][data-placeholder*='Type to customize this video' i], [contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i], [contenteditable='true'][aria-label*='Make a video' i], [contenteditable='true'][data-placeholder*='Customize video' i]");
                    if (!promptInput) return { ok: false, error: "Prompt input not found during verification" };
                    const value = promptInput.isContentEditable ? (promptInput.textContent || "") : (promptInput.value || "");
                    return { ok: !!value.trim(), filledLength: value.length };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        set_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const desiredQuality = "{selected_quality_label}";
                    const desiredAspect = "{selected_aspect_ratio}";
                    const desiredDuration = "{selected_duration_label}";
                    const fallbackQuality = desiredQuality === "720p" ? "480p" : desiredQuality;
                    const cleanText = (value) => String(value || "").replace(/\s+/g, " ").trim();

                    const menuCandidates = [
                        ...document.querySelectorAll("[role='menu'][data-state='open']"),
                        ...document.querySelectorAll("[data-radix-menu-content][data-state='open']"),
                        ...document.querySelectorAll("[role='menu'], [data-radix-menu-content]")
                    ];
                    const hasSoraItems = (el) => /\borientation\b|\bduration\b|\bvideos\b/i.test(cleanText(el?.textContent || ""));
                    const soraMenu = menuCandidates.find((el) => isVisible(el) && hasSoraItems(el)) || null;

                    if (soraMenu) {
                        const optionsRequested = [];
                        const optionsApplied = [];

                        const menuItems = (root) => [...root.querySelectorAll("[role='menuitem']")].filter((el) => isVisible(el));
                        const radioItems = (root) => [...root.querySelectorAll("[role='menuitemradio']")].filter((el) => isVisible(el));
                        const currentlyOpenMenus = () => [
                            ...document.querySelectorAll("[role='menu'][data-state='open']"),
                            ...document.querySelectorAll("[data-radix-menu-content][data-state='open']")
                        ].filter((el, idx, arr) => arr.indexOf(el) === idx && isVisible(el));

                        const findMenuItem = (labelPatterns) => {
                            const rows = menuItems(soraMenu);
                            return rows.find((el) => labelPatterns.some((pattern) => pattern.test(cleanText(el.textContent))));
                        };

                        const openSubmenu = (label, patterns) => {
                            const row = findMenuItem(patterns);
                            if (!row) return false;
                            const expanded = row.getAttribute("aria-expanded") === "true";
                            if (!expanded) {
                                if (!emulateClick(row)) return false;
                                optionsRequested.push(label);
                            }
                            optionsApplied.push(`${label}-menu-open`);
                            return true;
                        };

                        const pickRadioOption = (menuLabel, optionLabel, optionPatterns) => {
                            const openMenus = currentlyOpenMenus();
                            const radios = openMenus.flatMap((m) => radioItems(m));
                            const candidate = radios.find((el) => optionPatterns.some((pattern) => pattern.test(cleanText(el.textContent))));
                            if (!candidate) return false;
                            const checked = candidate.getAttribute("aria-checked") === "true" || (candidate.getAttribute("data-state") || "").toLowerCase() === "checked";
                            if (!checked) {
                                if (!emulateClick(candidate)) return false;
                                optionsRequested.push(`${menuLabel}:${optionLabel}`);
                            }
                            optionsApplied.push(`${menuLabel}:${optionLabel}`);
                            return true;
                        };

                        const desiredOrientation = ({
                            "9:16": "Portrait",
                            "2:3": "Portrait",
                            "16:9": "Landscape",
                            "3:2": "Landscape",
                            "1:1": "Square",
                        })[desiredAspect] || "Landscape";

                        const orientationPatterns = {
                            Portrait: [/^portrait$/i],
                            Landscape: [/^landscape$/i],
                            Square: [/^square$/i, /^1\s*:\s*1$/i],
                        };
                        const durationPatterns = {
                            "5s": [/^5\s*s(ec(onds?)?)?$/i],
                            "6s": [/^6\s*s(ec(onds?)?)?$/i],
                            "10s": [/^10\s*s(ec(onds?)?)?$/i],
                            "15s": [/^15\s*s(ec(onds?)?)?$/i],
                        };
                        const videoCountPatterns = {
                            "1 video": [/^1\s*video(s)?$/i],
                            "2 videos": [/^2\s*video(s)?$/i],
                        };

                        const desiredVideoCount = "1 video";

                        const orientationOpened = openSubmenu("orientation", [/\borientation\b/i]);
                        const orientationApplied = orientationOpened && pickRadioOption("orientation", desiredOrientation, orientationPatterns[desiredOrientation] || orientationPatterns.Landscape);

                        const durationOpened = openSubmenu("duration", [/\bduration\b/i]);
                        const durationApplied = durationOpened && (
                            pickRadioOption("duration", desiredDuration, durationPatterns[desiredDuration] || durationPatterns["10s"]) ||
                            pickRadioOption("duration", "10s", durationPatterns["10s"]) ||
                            pickRadioOption("duration", "15s", durationPatterns["15s"]) ||
                            pickRadioOption("duration", "5s", durationPatterns["5s"])
                        );

                        const videosOpened = openSubmenu("videos", [/\bvideos\b/i]);
                        const videosApplied = videosOpened && pickRadioOption("videos", desiredVideoCount, videoCountPatterns[desiredVideoCount]);

                        const requiredOptions = ["video", desiredDuration, desiredAspect, desiredVideoCount];
                        const missingOptions = [];
                        if (!orientationApplied) missingOptions.push(desiredAspect);
                        if (!durationApplied) missingOptions.push(desiredDuration);
                        if (!videosApplied) missingOptions.push(desiredVideoCount);

                        return {
                            ok: true,
                            requiredOptions,
                            effectiveRequiredOptions: requiredOptions,
                            optionsRequested,
                            optionsApplied,
                            missingOptions,
                            selectedQuality: "sora-default",
                            fallbackUsed: false,
                            soraMode: true,
                        };
                    }

                    const isOptionsMenu = (el) => {
                        if (!el || !isVisible(el)) return false;
                        const txt = cleanText(el.textContent || "");
                        return /video\s*duration|resolution|aspect\s*ratio|\bimage\b|\bvideo\b/i.test(txt);
                    };
                    const menu = menuCandidates.find((el) => isOptionsMenu(el)) || null;
                    if (!menu) return { ok: false, error: "Options menu not visible" };

                    const optionNodes = [...menu.querySelectorAll("button, [role='menuitemradio'], [role='radio'], [role='button']")]
                        .filter((el) => isVisible(el));

                    const nodeText = (el) => cleanText(el.getAttribute("aria-label") || el.textContent || "");
                    const isSelected = (el) => {
                        if (!el) return false;
                        const ariaChecked = el.getAttribute("aria-checked") === "true";
                        const ariaPressed = el.getAttribute("aria-pressed") === "true";
                        const ariaSelected = el.getAttribute("aria-selected") === "true";
                        const dataState = (el.getAttribute("data-state") || "").toLowerCase();
                        const dataSelected = el.getAttribute("data-selected") === "true";
                        const classSelected = /\b(bg-button-filled|active|selected|checked|font-semibold|text-primary)\b/i.test(el.className || "");
                        return ariaChecked || ariaPressed || ariaSelected || dataSelected || dataState === "checked" || dataState === "active" || classSelected;
                    };

                    const byExactName = (name) => optionNodes.find((el) => nodeText(el).toLowerCase() === String(name).toLowerCase()) || null;
                    const byPattern = (patterns) => optionNodes.find((el) => patterns.some((p) => p.test(nodeText(el)))) || null;

                    const qualityPatterns = {
                        "480p": [/^480p$/i, /480\s*p/i, /854\s*[x×]\s*480/i],
                        "720p": [/^720p$/i, /720\s*p/i, /1280\s*[x×]\s*720/i],
                    };
                    const durationPatterns = {
                        "6s": [/^6s$/i, /^6\s*s(ec(onds?)?)?$/i],
                        "10s": [/^10s$/i, /^10\s*s(ec(onds?)?)?$/i],
                    };
                    const aspectPatterns = {
                        "2:3": [/^2\s*:\s*3$/i],
                        "3:2": [/^3\s*:\s*2$/i],
                        "1:1": [/^1\s*:\s*1$/i],
                        "9:16": [/^9\s*:\s*16$/i],
                        "16:9": [/^16\s*:\s*9$/i],
                    };

                    const optionsRequested = [];
                    const optionsApplied = [];
                    let effectiveQuality = desiredQuality;

                    const chooseOption = (name, patterns) => {
                        const direct = byExactName(name);
                        const candidate = direct || byPattern(patterns);
                        if (!candidate) return false;
                        if (isSelected(candidate)) {
                            optionsApplied.push(`${name}(already-selected)`);
                            return true;
                        }
                        const clicked = emulateClick(candidate);
                        if (clicked) optionsRequested.push(name);
                        if (clicked && !isSelected(candidate)) emulateClick(candidate);
                        if (isSelected(candidate)) {
                            optionsApplied.push(name);
                            return true;
                        }
                        return false;
                    };

                    chooseOption("Video", [/^video$/i, /video\s*mode/i]);

                    const desiredQualityApplied = chooseOption(desiredQuality, qualityPatterns[desiredQuality] || qualityPatterns["720p"]);
                    if (!desiredQualityApplied && fallbackQuality !== desiredQuality) {
                        const fallbackApplied = chooseOption(fallbackQuality, qualityPatterns[fallbackQuality] || qualityPatterns["480p"]);
                        if (fallbackApplied) effectiveQuality = fallbackQuality;
                    }

                    chooseOption(desiredDuration, durationPatterns[desiredDuration] || durationPatterns["10s"]);
                    chooseOption(desiredAspect, aspectPatterns[desiredAspect] || aspectPatterns["16:9"]);

                    const selectedQualityNode = byExactName(effectiveQuality) || byPattern(qualityPatterns[effectiveQuality] || qualityPatterns["720p"]);
                    if (!isSelected(selectedQualityNode) && fallbackQuality !== desiredQuality) {
                        const fallbackNode = byExactName(fallbackQuality) || byPattern(qualityPatterns[fallbackQuality] || qualityPatterns["480p"]);
                        if (isSelected(fallbackNode)) effectiveQuality = fallbackQuality;
                    }

                    const requiredOptions = ["video", desiredQuality, desiredDuration, desiredAspect];
                    const effectiveRequiredOptions = ["video", effectiveQuality, desiredDuration, desiredAspect];

                    const isOptionConfirmed = (name) => {
                        if (String(name).toLowerCase() === "video") {
                            const videoNode = byExactName("Video") || byPattern([/^video$/i]);
                            return isSelected(videoNode);
                        }
                        if (name === effectiveQuality || name === desiredQuality) {
                            const n = byExactName(name) || byPattern(qualityPatterns[name] || qualityPatterns["720p"]);
                            return isSelected(n);
                        }
                        if (name === desiredDuration) {
                            const n = byExactName(name) || byPattern(durationPatterns[name] || durationPatterns["10s"]);
                            return isSelected(n);
                        }
                        const n = byExactName(name) || byPattern(aspectPatterns[name] || aspectPatterns["16:9"]);
                        return isSelected(n);
                    };

                    const missingOptions = effectiveRequiredOptions.filter((name) => !isOptionConfirmed(name));
                    return {
                        ok: true,
                        requiredOptions,
                        effectiveRequiredOptions,
                        optionsRequested,
                        optionsApplied,
                        missingOptions,
                        selectedQuality: effectiveQuality,
                        fallbackUsed: effectiveQuality !== desiredQuality,
                        soraMode: false,
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """
        set_options_script = set_options_script.replace('"{selected_quality_label}"', json.dumps(selected_quality_label))
        set_options_script = set_options_script.replace('"{selected_duration_label}"', json.dumps(selected_duration_label))
        set_options_script = set_options_script.replace('"{selected_aspect_ratio}"', json.dumps(selected_aspect_ratio))

        close_options_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        if (!el || !isVisible(el) || el.disabled) return false;
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    };

                    const isOptionsMenu = (el) => {
                        if (!el || !isVisible(el)) return false;
                        const txt = (el.textContent || "").trim();
                        return /video\s*duration|resolution|aspect\s*ratio|\borientation\b|\bduration\b|\bvideos\b|\bimage\b|\bvideo\b/i.test(txt);
                    };
                    const hasOpenMenu = () => {
                        const candidates = [
                            ...document.querySelectorAll("[role='menu'][data-state='open']"),
                            ...document.querySelectorAll("[data-radix-menu-content][data-state='open']"),
                            ...document.querySelectorAll("[role='menu'], [data-radix-menu-content]")
                        ];
                        return candidates.some((el) => isOptionsMenu(el));
                    };

                    let closed = false;
                    const settingsButton = [...document.querySelectorAll("button[aria-label='Settings']")[1]]
                        .find((el) => isVisible(el) && !el.disabled);
                    if (settingsButton) closed = emulateClick(settingsButton);

                    if (hasOpenMenu()) {
                        const videoChip = [...document.querySelectorAll("button[aria-label='Video']")]
                            .find((el) => isVisible(el) && !el.disabled && !el.closest("[role='menu']") && !el.closest("[data-radix-menu-content]"));
                        if (videoChip) closed = emulateClick(videoChip) || closed;
                    }

                    if (hasOpenMenu()) {
                        const escEvent = new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true });
                        document.dispatchEvent(escEvent);
                        closed = true;
                    }

                    return { ok: true, closed, panelVisible: hasOpenMenu() };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        submit_script = r"""
            (() => {
                try {
                    const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                    const promptInput = document.querySelector("textarea[placeholder*='Describe your video' i], textarea[aria-label*='Describe your video' i], textarea[placeholder*='Type to imagine' i], input[placeholder*='Type to imagine' i], textarea[placeholder*='Type to customize this video' i], input[placeholder*='Type to customize this video' i], textarea[placeholder*='Type to customize video' i], input[placeholder*='Type to customize video' i], textarea[placeholder*='Customize video' i], input[placeholder*='Customize video' i], textarea[aria-label*='Make a video' i], input[aria-label*='Make a video' i], div.tiptap.ProseMirror[contenteditable='true'], [contenteditable='true'][aria-label*='Type to imagine' i], [contenteditable='true'][data-placeholder*='Type to imagine' i], [contenteditable='true'][aria-label*='Type to customize this video' i], [contenteditable='true'][data-placeholder*='Type to customize this video' i], [contenteditable='true'][aria-label*='Type to customize video' i], [contenteditable='true'][data-placeholder*='Type to customize video' i], [contenteditable='true'][aria-label*='Make a video' i], [contenteditable='true'][data-placeholder*='Customize video' i]");

                    const submitSelectors = [
                        "button[type='submit'][aria-label='Submit']",
                        "button[aria-label='Submit'][type='submit']",
                        "button[type='submit']",
                        "button[aria-label='Submit']",
                        "button[aria-label='Create video']",
                        "button[aria-label*='Create video' i]",
                        "button[data-disabled='false']"
                    ];

                    const submitCandidates = [];
                    const collect = (root) => {
                        if (!root || typeof root.querySelectorAll !== "function") return;
                        submitSelectors.forEach((selector) => {
                            const matches = root.querySelectorAll(selector);
                            for (let i = 0; i < matches.length; i += 1) submitCandidates.push(matches[i]);
                        });
                    };

                    const composerRoot = (promptInput && typeof promptInput.closest === "function")
                        ? (promptInput.closest("form") || promptInput.closest("main") || promptInput.closest("section") || promptInput.parentElement)
                        : null;

                    collect(composerRoot);
                    collect(document);

                    const uniqueCandidates = [...new Set(submitCandidates)];
                    const submitButton = uniqueCandidates.find((el) => isVisible(el));

                    const form = (submitButton && submitButton.form)
                        || (promptInput && typeof promptInput.closest === "function" ? promptInput.closest("form") : null)
                        || (composerRoot && typeof composerRoot.closest === "function" ? composerRoot.closest("form") : null)
                        || document.querySelector("form");

                    if (!submitButton && !form) return { ok: false, error: "Submit button/form not found" };

                    if (submitButton && submitButton.disabled) {
                        submitButton.disabled = false;
                        submitButton.removeAttribute("disabled");
                        submitButton.setAttribute("aria-disabled", "false");
                    }

                    const common = { bubbles: true, cancelable: true, composed: true };
                    const emulateClick = (el) => {
                        try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                    };

                    const allButtons = [...document.querySelectorAll("button")].filter((el) => isVisible(el));
                    const createVideoButton = allButtons.find((btn) => {
                        const label = (btn.getAttribute("aria-label") || "").trim();
                        const txt = (btn.textContent || "").trim();
                        const srOnly = (btn.querySelector(".sr-only")?.textContent || "").trim();
                        return /create\s*video/i.test(label) || /create\s*video/i.test(txt) || /create\s*video/i.test(srOnly);
                    }) || submitButton;

                    let clicked = false;
                    if (createVideoButton) {
                        emulateClick(createVideoButton);
                        clicked = true;
                    }

                    let formSubmitted = false;
                    if (!clicked && form) {
                        const ev = new Event("submit", { bubbles: true, cancelable: true });
                        form.dispatchEvent(ev); // lets React handlers run
                        formSubmitted = true;
                    }

                    return {
                        ok: clicked || formSubmitted,
                        submitted: clicked || formSubmitted,
                        doubleClicked: !!createVideoButton,
                        formSubmitted,
                        forceEnabled: !!createVideoButton,
                        buttonText: createVideoButton ? (createVideoButton.textContent || "").trim() : "",
                        buttonAriaLabel: createVideoButton ? (createVideoButton.getAttribute("aria-label") || "") : ""
                    };
                } catch (err) {
                    return { ok: false, error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """

        def _continue_after_options_set(result):
            if not isinstance(result, dict) or not result.get("ok"):
                options_error = result.get("error") if isinstance(result, dict) else result
                self._append_log(
                    f"WARNING: Option application script reported an error for variant {variant}: {options_error!r}. Continuing."
                )

            options_requested = result.get("optionsRequested") if isinstance(result, dict) else []
            options_applied = result.get("optionsApplied") if isinstance(result, dict) else []
            selected_quality = result.get("selectedQuality") if isinstance(result, dict) else None
            fallback_used = bool(result.get("fallbackUsed")) if isinstance(result, dict) else False
            requested_summary = ", ".join(options_requested) if options_requested else "none"
            applied_summary = ", ".join(options_applied) if options_applied else "none detected"
            self._append_log(
                f"Options staged for variant {variant}; options requested: {requested_summary}; options applied markers: {applied_summary}."
            )
            if selected_quality:
                if fallback_used:
                    self._append_log(
                        f"Variant {variant}: preferred quality {selected_quality_label} not confirmed in the panel; "
                        f"falling back to {selected_quality}."
                    )
                else:
                    self._append_log(f"Variant {variant}: confirmed quality selection {selected_quality}.")


            def _continue_after_options_close(close_result):
                if not isinstance(close_result, dict) or not close_result.get("ok"):
                    close_error = close_result.get("error") if isinstance(close_result, dict) else close_result
                    self._append_log(
                        f"WARNING: Closing options window reported an error for variant {variant}: {close_error!r}. Continuing."
                    )

                self._append_log(f"Options window closed for variant {variant}; submitting after {action_delay_ms}ms delay.")

                def after_delayed_submit(submit_result):
                    if not isinstance(submit_result, dict) or not submit_result.get("ok"):
                        error_detail = submit_result.get("error") if isinstance(submit_result, dict) else submit_result
                        self._append_log(
                            f"WARNING: Manual submit script reported an issue for variant {variant}: {error_detail!r}. Continuing to download polling."
                        )

                    self._append_log(
                        "Submitted manual variant "
                        f"{variant} after prompt/options staged delays (double-click submit); "
                        "waiting for generation to auto-download."
                    )
                    self._trigger_browser_video_download(variant)

                QTimer.singleShot(action_delay_ms, lambda: self.browser.page().runJavaScript(submit_script, after_delayed_submit))

            QTimer.singleShot(
                action_delay_ms,
                lambda: self.browser.page().runJavaScript(close_options_script, _continue_after_options_close),
            )

        def _continue_after_options_open(open_result):
            open_ok = isinstance(open_result, dict) and bool(open_result.get("ok"))
            panel_visible = isinstance(open_result, dict) and bool(open_result.get("panelVisible"))
            if not open_ok:
                open_error = open_result.get("error") if isinstance(open_result, dict) else open_result
                self._append_log(
                    f"WARNING: Opening options window returned an error for variant {variant}: {open_error!r}. "
                    "Continuing anyway."
                )
            elif panel_visible:
                self._append_log(f"Options panel appears visible for variant {variant}; proceeding to option selection.")

            self._append_log(f"Options window opened for variant {variant}; setting options after {action_delay_ms}ms delay.")
            QTimer.singleShot(
                action_delay_ms,
                lambda: self.browser.page().runJavaScript(set_options_script, _continue_after_options_set),
            )

        def _run_continue_mode_submit() -> None:
            continue_submit_script = r"""
                (() => {
                    try {
                        const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                        const common = { bubbles: true, cancelable: true, composed: true };
                        const emulateClick = (el) => {
                            if (!el || !isVisible(el) || el.disabled) return false;
                            try { el.dispatchEvent(new PointerEvent("pointerdown", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mousedown", common));
                            try { el.dispatchEvent(new PointerEvent("pointerup", common)); } catch (_) {}
                            el.dispatchEvent(new MouseEvent("mouseup", common));
                            el.dispatchEvent(new MouseEvent("click", common));
                            return true;
                        };

                        const buttons = [...document.querySelectorAll("button, [role='button']")].filter((el) => isVisible(el));
                        const matchers = [
                            /make\s*video/i,
                            /generate/i,
                            /submit/i,
                        ];
                        let actionButton = null;
                        for (const matcher of matchers) {
                            actionButton = buttons.find((btn) => matcher.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));
                            if (actionButton) break;
                        }
                        const clicked = actionButton ? emulateClick(actionButton) : false;
                        return {
                            ok: clicked,
                            buttonText: actionButton ? (actionButton.textContent || "").trim() : "",
                            buttonAriaLabel: actionButton ? (actionButton.getAttribute("aria-label") || "") : "",
                            error: clicked ? "" : "Could not find/click Make video button",
                        };
                    } catch (err) {
                        return { ok: false, error: String(err && err.stack ? err.stack : err) };
                    }
                })()
            """

            def _after_continue_submit(submit_result):
                if not isinstance(submit_result, dict) or not submit_result.get("ok"):
                    error_detail = submit_result.get("error") if isinstance(submit_result, dict) else submit_result
                    if error_detail not in (None, "", "callback-empty"):
                        self._append_log(
                            f"Continue-mode submit for variant {variant} reported an issue: {error_detail!r}; continuing to video download polling."
                        )
                else:
                    button_label = submit_result.get("buttonAriaLabel") or submit_result.get("buttonText") or "Make video"
                    self._append_log(f"Continue-mode submit clicked '{button_label}' for variant {variant}.")
                self._trigger_browser_video_download(variant, allow_make_video_click=False)

            QTimer.singleShot(action_delay_ms, lambda: self.browser.page().runJavaScript(continue_submit_script, _after_continue_submit))

        def after_submit(result):
            fill_ok = isinstance(result, dict) and bool(result.get("ok"))
            if not fill_ok:
                error_detail = result.get("error") if isinstance(result, dict) else result
                if error_detail not in (None, "", "callback-empty"):
                    self._append_log(
                        f"Manual prompt fill reported an issue for variant {variant}: {error_detail!r}. "
                        "Verifying current field content before continuing."
                    )

            def _after_verify_prompt(verify_result):
                verify_ok = isinstance(verify_result, dict) and bool(verify_result.get("ok"))
                if not (fill_ok or verify_ok):
                    verify_error = verify_result.get("error") if isinstance(verify_result, dict) else verify_result
                    if verify_error not in (None, "", "callback-empty"):
                        self._append_log(
                            f"Prompt fill verification did not confirm content for variant {variant}: {verify_error!r}. "
                            "Continuing with option selection and forced submit anyway."
                        )
                if self.continue_from_frame_active:
                    _run_continue_mode_submit()
                else:
                    QTimer.singleShot(
                        action_delay_ms,
                        lambda: self.browser.page().runJavaScript(open_options_script, _continue_after_options_open),
                    )

            if fill_ok:
                _after_verify_prompt({"ok": True})
                return

            QTimer.singleShot(
                250,
                lambda: self.browser.page().runJavaScript(verify_prompt_script, _after_verify_prompt),
            )

        def _after_direct_post(post_result):
            post_status = post_result.get("status") if isinstance(post_result, dict) else "unknown"
            response_snippet = post_result.get("responseSnippet") if isinstance(post_result, dict) else ""
            self._append_log(
                f"Variant {variant}: /conversations/new response status={post_status}; body snippet={response_snippet!r}"
            )
            if isinstance(post_result, dict) and post_result.get("ok"):
                payload_summary = post_result.get("payloadSummary") or {}
                self._append_log(
                    f"Submitted manual variant {variant} via direct POST "
                    f"(aspect={payload_summary.get('aspectRatio')}, duration={payload_summary.get('videoLength')}s, "
                    f"resolution={payload_summary.get('resolutionName')}). Waiting 2.0s before continuing."
                )
                QTimer.singleShot(2000, lambda: self._trigger_browser_video_download(variant))
                return

            post_error = post_result.get("error") if isinstance(post_result, dict) else post_result
            self._append_log(
                f"WARNING: Direct POST submit failed for variant {variant}: {post_error!r}. "
                "Waiting 2.0s before falling back to browser UI automation."
            )
            QTimer.singleShot(2000, lambda: self.browser.page().runJavaScript(script, after_submit))

        self.browser.page().runJavaScript(post_create_video_script, _after_direct_post)

    def _trigger_browser_video_download(self, variant: int, allow_make_video_click: bool = True) -> None:
        self.pending_manual_download_type = "video"
        self.manual_download_deadline = time.time() + 420
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = allow_make_video_click
        self.manual_download_in_progress = False
        self.manual_download_started_at = time.time()
        self.manual_download_poll_timer.start(0)

    def _poll_for_manual_video(self) -> None:
        if self.stop_all_requested:
            self._append_log("Stop-all flag active; skipping queued job activity.")
            return

        variant = self.pending_manual_variant_for_download
        if variant is None:
            return

        deadline = self.manual_download_deadline or 0
        if time.time() > deadline:
            self.pending_manual_variant_for_download = None
            self.manual_download_click_sent = False
            self.manual_download_request_pending = False
            self.manual_video_start_click_sent = False
            self.manual_video_make_click_fallback_used = False
            self.manual_video_allow_make_click = True
            self.manual_download_in_progress = False
            self.manual_download_started_at = None
            self.manual_download_deadline = None
            self._append_log(f"ERROR: Variant {variant} did not produce a downloadable video in time.")
            if self.continue_from_frame_active:
                self._append_log("Continue-from-last-frame stopped because download polling timed out.")
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
            return

        allow_make_video_click = "true" if (self.manual_video_allow_make_click and not self.manual_video_start_click_sent) else "false"
        script = f"""
            (() => {{
                const allowMakeVideoClick = {allow_make_video_click};
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const common = {{ bubbles: true, cancelable: true, composed: true }};
                const emulateClick = (el) => {{
                    if (!el || !isVisible(el) || el.disabled) return false;
                    try {{
                        el.dispatchEvent(new PointerEvent("pointerdown", common));
                        el.dispatchEvent(new MouseEvent("mousedown", common));
                        el.dispatchEvent(new PointerEvent("pointerup", common));
                        el.dispatchEvent(new MouseEvent("mouseup", common));
                        el.dispatchEvent(new MouseEvent("click", common));
                        return true;
                    }} catch (_) {{
                        try {{
                            el.click();
                            return true;
                        }} catch (__){{
                            return false;
                        }}
                    }}
                }};
                const percentNode = [...document.querySelectorAll("div .tabular-nums, div.tabular-nums")]
                    .find((el) => isVisible(el) && /^\\d{{1,3}}%$/.test((el.textContent || "").trim()));
                if (percentNode) {{
                    return {{ status: "progress", progressText: (percentNode.textContent || "").trim() }};
                }}

                const cancelVideoButton = [...document.querySelectorAll("button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /cancel\\s+video/i.test((btn.getAttribute("aria-label") || btn.textContent || "").trim()));

                const redoButton = [...document.querySelectorAll("button")]
                    .find((btn) => isVisible(btn) && !btn.disabled && /redo/i.test((btn.textContent || "").trim()));

                const makeVideoButton = [...document.querySelectorAll("button")]
                    .find((btn) => {{
                        if (!isVisible(btn) || btn.disabled) return false;
                        const label = (btn.getAttribute("aria-label") || btn.textContent || "").trim();
                        return /make\\s+video/i.test(label);
                    }});

                const video = document.querySelector("video");
                const source = document.querySelector("video source");
                const src = (video && (video.currentSrc || video.src)) || (source && source.src) || "";
                const videoAnchor = video ? video.closest("a[href]") : null;
                const anchorHref = videoAnchor ? (videoAnchor.getAttribute("href") || "").trim() : "";
                const anchorUrl = (() => {{
                    if (!anchorHref) return "";
                    try {{
                        return new URL(anchorHref, window.location.href).toString();
                    }} catch (_) {{
                        return anchorHref;
                    }}
                }})();

                const isDirectVideoUrl = (url) => {{
                    if (!url) return false;
                    const normalized = String(url || "").trim();
                    if (!/^https?:[/][/]/i.test(normalized)) return false;
                    const isOpenAIVideo = /^https:[/][/]videos[.]openai[.]com[/]/i.test(normalized) && /[/]raw(?:$|[?#])/i.test(normalized);
                    const isImaginePublicVideo = /^https:[/][/]imagine-public[.]x[.]ai[/]/i.test(normalized) && /[.]mp4(?:$|[?#])/i.test(normalized);
                    return isOpenAIVideo || isImaginePublicVideo;
                }};

                if (isDirectVideoUrl(anchorUrl)) {{
                    return {{ status: "direct-url-ready", src: anchorUrl, sourceType: "video-anchor" }};
                }}
                if (isDirectVideoUrl(src)) {{
                    return {{ status: "direct-url-ready", src, sourceType: "video-src" }};
                }}

                const exactDownloadSelector = "button[type='button'][aria-label='Download']";
                const exactDownloadCandidates = [...document.querySelectorAll(exactDownloadSelector)]
                    .filter((btn) => isVisible(btn) && !btn.disabled);
                const fallbackDownloadCandidates = [...document.querySelectorAll("button[aria-label='Download']")]
                    .filter((btn) => isVisible(btn) && !btn.disabled);
                const downloadCandidates = [...exactDownloadCandidates, ...fallbackDownloadCandidates]
                    .filter((btn, index, arr) => arr.indexOf(btn) === index);
                const makeVideoContainer = makeVideoButton
                    ? (makeVideoButton.closest("form") || makeVideoButton.closest("section") || makeVideoButton.closest("main") || makeVideoButton.parentElement)
                    : null;
                let downloadButton = exactDownloadCandidates[0] || null;
                if (!downloadButton) {{
                    downloadButton = downloadCandidates.find((btn) => makeVideoContainer && makeVideoContainer.contains(btn) && btn !== makeVideoButton);
                }}
                if (!downloadButton) downloadButton = downloadCandidates[0] || null;

                if (downloadButton && !cancelVideoButton) {{
                    return {{
                        status: emulateClick(downloadButton) ? "download-clicked" : "download-visible",
                    }};
                }}

                if (cancelVideoButton) {{
                    return {{ status: "rendering-cancel-visible" }};
                }}

                if (makeVideoButton) {{
                    const buttonLabel = (makeVideoButton.getAttribute("aria-label") || makeVideoButton.textContent || "").trim();
                    if (!allowMakeVideoClick) {{
                        return {{ status: "make-video-awaiting-progress", buttonLabel }};
                    }}
                    return {{
                        status: emulateClick(makeVideoButton) ? "make-video-clicked" : "make-video-visible",
                        buttonLabel,
                    }};
                }}

                if (!redoButton) {{
                    return {{ status: "waiting-for-download" }};
                }}

                const enoughData = !!(video && video.readyState >= 3 && Number(video.duration || 0) > 0);
                return {{
                    status: src ? (enoughData ? "video-src-ready" : "video-buffering") : "waiting",
                    src,
                    readyState: video ? video.readyState : 0,
                    duration: video ? Number(video.duration || 0) : 0,
                }};
            }})()
        """

        def after_poll(result):
            current_variant = self.pending_manual_variant_for_download
            if current_variant is None:
                return

            if not isinstance(result, dict):
                self.manual_download_poll_timer.start(3000)
                return

            status = result.get("status", "waiting")
            progress_text = (result.get("progressText") or "").strip()

            if status == "progress":
                self.manual_video_start_click_sent = True
                if progress_text:
                    self._append_log(f"Variant {current_variant} still rendering: {progress_text}")
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-clicked":
                label = (result.get("buttonLabel") or "Make video").strip()
                self._append_log(f"Variant {current_variant}: clicked '{label}' to start video generation.")
                self.manual_video_start_click_sent = True
                self.manual_video_make_click_fallback_used = True
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-awaiting-progress":
                self.manual_download_poll_timer.start(3000)
                return

            if status == "make-video-visible":
                self._append_log(f"Variant {current_variant}: '{result.get('buttonLabel') or 'Make video'}' is visible but click did not register; retrying.")
                self.manual_download_poll_timer.start(2000)
                return

            if status in ("waiting-for-redo", "waiting-for-download", "rendering-cancel-visible"):
                self.manual_video_start_click_sent = True
                self.manual_download_poll_timer.start(3000)
                return

            if status == "download-clicked":
                if not self.manual_download_click_sent:
                    self._append_log(f"Variant {current_variant} appears ready; clicked in-page Download button.")
                    self.manual_download_click_sent = True
                    self.manual_download_in_progress = True
                self.manual_download_poll_timer.start(3000)
                return

            src = result.get("src") or ""
            if status == "video-buffering":
                self.manual_download_poll_timer.start(3000)
                return

            min_wait_elapsed = self.manual_download_started_at is not None and (time.time() - self.manual_download_started_at) >= 8
            if status not in ("video-src-ready", "direct-url-ready") or not src or not min_wait_elapsed:
                self.manual_download_poll_timer.start(3000)
                return

            if status == "direct-url-ready":
                if self.manual_download_click_sent:
                    self.manual_download_poll_timer.start(3000)
                    return
                source_type = result.get("sourceType") or "video-link"
                self._append_log(f"Variant {current_variant} ready; downloading directly from detected video URL ({source_type}).")
                try:
                    self.download_dir.mkdir(parents=True, exist_ok=True)
                    filename = self._build_session_download_filename("video", current_variant, "mp4")
                    video_path = self.download_dir / filename
                    with requests.get(src, stream=True, timeout=240) as response:
                        response.raise_for_status()
                        with open(video_path, "wb") as handle:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    handle.write(chunk)
                    video_size = video_path.stat().st_size if video_path.exists() else 0
                    if video_size < MIN_VALID_VIDEO_BYTES:
                        self._append_log(
                            f"WARNING: Direct URL download for variant {current_variant} is only {video_size} bytes (< 1MB); retrying browser download flow."
                        )
                        if video_path.exists():
                            video_path.unlink(missing_ok=True)
                        self.manual_download_poll_timer.start(2000)
                        return
                    self.manual_download_click_sent = True
                    self._complete_manual_video_download(video_path, current_variant)
                    return
                except Exception as exc:
                    self._append_log(f"WARNING: Direct URL download failed for variant {current_variant}: {exc}")
                    self.manual_download_poll_timer.start(3000)
                    return

            trigger_download_script = f"""
                (() => {{
                    const src = {src!r};
                    const a = document.createElement("a");
                    a.href = src;
                    a.download = `grok_manual_variant_{current_variant}_${{Date.now()}}.mp4`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    return true;
                }})()
            """
            if not self.manual_download_click_sent:
                self.browser.page().runJavaScript(trigger_download_script)
                self._append_log(f"Variant {current_variant} video detected; browser download requested from video source.")
                self.manual_download_click_sent = True
                self.manual_download_in_progress = True

            self.manual_download_poll_timer.start(3000)

        self.browser.page().runJavaScript(script, after_poll)

    def _complete_manual_video_download(self, video_path: Path, variant: int) -> None:
        self._advance_automation_counter_tracking()
        self.on_video_finished(
            {
                "title": f"Manual Browser Video {variant}",
                "prompt": self.manual_prompt.toPlainText().strip(),
                "resolution": "web",
                "video_file_path": str(video_path),
                "source_url": "browser-session",
            }
        )
        self._return_embedded_browser_after_download()
        self.pending_manual_variant_for_download = None
        self.pending_manual_download_type = None
        self.pending_manual_image_prompt = None
        self.pending_manual_redirect_target = "grok"
        self.manual_image_pick_clicked = False
        self.manual_image_video_mode_selected = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_video_mode_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = True
        self.manual_download_in_progress = False
        self.manual_download_started_at = None
        self.manual_download_deadline = None
        if self.continue_from_frame_active:
            self.continue_from_frame_completed += 1
            if self.continue_from_frame_completed < self.continue_from_frame_target_count:
                QTimer.singleShot(800, self._start_continue_iteration)
            else:
                self._append_log("Continue workflow complete.")
                self._reset_automation_counter_tracking()
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
                self.continue_from_frame_current_source_video = ""
                self.continue_from_frame_seed_image_path = None
        else:
            self._submit_next_manual_variant()

    def _on_browser_download_requested(self, download) -> None:
        variant = self.pending_manual_variant_for_download
        if variant is None:
            return
        if self.manual_download_request_pending:
            self._append_log("Ignoring duplicate browser download request for current manual variant.")
            download.cancel()
            return
        if self.manual_download_in_progress:
            self.manual_download_in_progress = False
        elif self.manual_download_click_sent:
            self._append_log("Ignoring duplicate browser download request for current manual variant.")
            download.cancel()
            return

        download_type = self.pending_manual_download_type or "video"
        extension = self._resolve_download_extension(download, download_type)
        filename = self._build_session_download_filename(download_type, variant, extension)
        download.setDownloadDirectory(str(self.download_dir))
        download.setDownloadFileName(filename)
        self.manual_download_click_sent = True
        self.manual_download_request_pending = True
        download.accept()
        self._append_log(f"Downloading manual {download_type} variant {variant} to {self.download_dir / filename}")

        def on_state_changed(state):
            if state == download.DownloadState.DownloadCompleted:
                video_path = self.download_dir / filename
                video_size = video_path.stat().st_size if video_path.exists() else 0
                if download_type == "image":
                    self._append_log(f"Saved image: {video_path}")
                    self.pending_manual_variant_for_download = None
                    self.pending_manual_download_type = None
                    self.pending_manual_image_prompt = None
                    self.manual_image_pick_clicked = False
                    self.manual_image_video_mode_selected = False
                    self.manual_image_video_submit_sent = False
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_video_mode_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    self.manual_download_click_sent = False
                    self.manual_download_request_pending = False
                    self.manual_video_start_click_sent = False
                    self.manual_video_make_click_fallback_used = False
                    self.manual_video_allow_make_click = True
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = None
                    self.manual_download_deadline = None
                    self._submit_next_manual_image_variant()
                    return

                image_extensions = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
                if download_type == "video" and extension.lower() in image_extensions:
                    self._append_log(
                        f"WARNING: Variant {variant}: clicked a non-video download target ({extension}); retrying with video download button."
                    )
                    if video_path.exists():
                        video_path.unlink(missing_ok=True)
                    self.manual_download_click_sent = False
                    self.manual_download_request_pending = False
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = time.time()
                    self.manual_download_poll_timer.start(1200)
                    return

                if video_size < MIN_VALID_VIDEO_BYTES:
                    self._append_log(
                        f"WARNING: Downloaded manual variant {variant} is only {video_size} bytes (< 1MB)."
                    )
                    self.pending_manual_variant_for_download = None
                    self.pending_manual_download_type = None
                    self.pending_manual_image_prompt = None
                    self.manual_image_pick_clicked = False
                    self.manual_image_video_mode_selected = False
                    self.manual_image_video_submit_sent = False
                    self.manual_image_pick_retry_count = 0
                    self.manual_image_video_mode_retry_count = 0
                    self.manual_image_submit_retry_count = 0
                    self.manual_download_click_sent = False
                    self.manual_download_request_pending = False
                    self.manual_video_start_click_sent = False
                    self.manual_video_make_click_fallback_used = False
                    self.manual_video_allow_make_click = True
                    self.manual_download_in_progress = False
                    self.manual_download_started_at = None
                    self.manual_download_deadline = None

                    if self.continue_from_frame_active:
                        self._retry_continue_after_small_download(variant)
                    else:
                        self._append_log(
                            "WARNING: Undersized manual download detected outside continue-from-last-frame mode; "
                            "please use 'Continue from Last Frame' to regenerate from the extracted frame."
                        )
                    return

                self._complete_manual_video_download(video_path, variant)
            elif state == download.DownloadState.DownloadInterrupted:
                self._append_log(f"ERROR: Download interrupted for manual variant {variant}.")
                self.pending_manual_variant_for_download = None
                self.pending_manual_download_type = None
                self.pending_manual_image_prompt = None
                self.manual_image_pick_clicked = False
                self.manual_image_video_mode_selected = False
                self.manual_image_video_submit_sent = False
                self.manual_image_pick_retry_count = 0
                self.manual_image_video_mode_retry_count = 0
                self.manual_image_submit_retry_count = 0
                self.manual_download_click_sent = False
                self.manual_download_request_pending = False
                self.manual_video_start_click_sent = False
                self.manual_video_make_click_fallback_used = False
                self.manual_video_allow_make_click = True
                self.manual_download_in_progress = False
                self.manual_download_started_at = None
                self.manual_download_deadline = None
                self.continue_from_frame_active = False
                self.continue_from_frame_target_count = 0
                self.continue_from_frame_completed = 0
                self.continue_from_frame_prompt = ""
                self.continue_from_frame_current_source_video = ""
                self.continue_from_frame_seed_image_path = None

        download.stateChanged.connect(on_state_changed)

    def stop_all_jobs(self) -> None:
        self.stop_all_requested = True
        self.manual_generation_queue.clear()
        self.manual_image_generation_queue.clear()
        self.manual_download_poll_timer.stop()
        self.continue_from_frame_reload_timeout_timer.stop()
        self.pending_manual_variant_for_download = None
        self.pending_manual_download_type = None
        self.pending_manual_image_prompt = None
        self.manual_image_pick_clicked = False
        self.manual_image_video_mode_selected = False
        self.manual_image_video_submit_sent = False
        self.manual_image_pick_retry_count = 0
        self.manual_image_video_mode_retry_count = 0
        self.manual_image_submit_retry_count = 0
        self.manual_image_submit_token += 1
        self.manual_download_click_sent = False
        self.manual_download_request_pending = False
        self.manual_video_start_click_sent = False
        self.manual_video_make_click_fallback_used = False
        self.manual_video_allow_make_click = True
        self.manual_download_in_progress = False
        self.manual_download_started_at = None
        self.manual_download_deadline = None
        self._reset_automation_counter_tracking()

        self.continue_from_frame_active = False
        self.continue_from_frame_target_count = 0
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = ""
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = None
        self.continue_from_frame_waiting_for_reload = False

        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append_log("Stop requested: API generation worker will stop after the current request completes.")
        if self.stitch_worker and self.stitch_worker.isRunning():
            self.stitch_worker.request_stop()
            self._append_log("Stop requested: stitching/encoding worker is being interrupted.")
        if self.udp_workflow_worker and self.udp_workflow_worker.isRunning():
            try:
                self.udp_workflow_worker.request_stop()
            except Exception:
                pass
            self._append_automation_log("Stop requested: active UDP automation workflow interrupted.")
        if self._active_ffmpeg_process is not None and self._active_ffmpeg_process.poll() is None:
            self._active_ffmpeg_process.terminate()
            self._append_log("Stop requested: active ffmpeg stitch/encode process terminated.")
        self._append_log("Stop all requested: cleared queued manual image/video jobs and halted polling timers.")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop_all_jobs()
        try:
            self._cdp_relay_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if self.automation_runtime is not None:
            try:
                self.automation_runtime.stop_runtime()
            except Exception:
                pass
            if self.automation_runtime.isRunning():
                self.automation_runtime.quit()
                self.automation_runtime.wait(3000)
        workers: list[tuple[str, QThread | None]] = [
            ("generation", self.worker),
            ("stitch", self.stitch_worker),
            ("upload", self.upload_worker),
            ("browser-training", self.browser_training_worker),
            ("overlay", self.overlay_worker),
        ]
        for worker_name, worker in workers:
            if worker is None or not worker.isRunning():
                continue
            if hasattr(worker, "request_stop"):
                try:
                    worker.request_stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
            worker.requestInterruption()
            if not worker.wait(5000):
                self._append_log(f"WARNING: Timed out while waiting for {worker_name} worker thread to stop.")
        super().closeEvent(event)

    def _build_video_label(self, video: dict) -> str:
        title = str(video.get("title") or "Video")
        resolution = str(video.get("resolution") or "unknown")
        return f"{title} ({resolution})"

    def _serialize_video_list_for_preferences(self) -> list[dict[str, str]]:
        serialized: list[dict[str, str]] = []
        for video in self.videos:
            if not isinstance(video, dict):
                continue
            video_path = str(video.get("video_file_path") or "").strip()
            if not video_path:
                continue
            serialized.append(
                {
                    "title": str(video.get("title") or "Video"),
                    "prompt": str(video.get("prompt") or ""),
                    "resolution": str(video.get("resolution") or "unknown"),
                    "video_file_path": video_path,
                    "source_url": str(video.get("source_url") or ""),
                }
            )
        return serialized

    def _restore_video_list_from_preferences(self, saved_videos: object, *, selected_index: object = None) -> None:
        restored_videos: list[dict] = []
        missing_paths: list[str] = []

        if isinstance(saved_videos, list):
            for item in saved_videos:
                if not isinstance(item, dict):
                    continue
                raw_path = str(item.get("video_file_path") or "").strip()
                if not raw_path:
                    continue

                video_path = Path(raw_path).expanduser()
                if not video_path.is_absolute():
                    video_path = (BASE_DIR / video_path).resolve()
                if not video_path.exists():
                    missing_paths.append(str(video_path))
                    continue

                restored_videos.append(
                    {
                        "title": str(item.get("title") or video_path.stem),
                        "prompt": str(item.get("prompt") or ""),
                        "resolution": str(item.get("resolution") or "unknown"),
                        "video_file_path": str(video_path),
                        "source_url": str(item.get("source_url") or "preferences"),
                    }
                )

        self.videos = restored_videos
        if not self.videos:
            self._refresh_video_picker(selected_index=-1)
            return

        target_index = len(self.videos) - 1
        try:
            selected = int(selected_index)
            if 0 <= selected < len(self.videos):
                target_index = selected
        except (TypeError, ValueError):
            pass

        self._refresh_video_picker(selected_index=target_index)

        if missing_paths:
            self._append_log(
                "Skipped missing saved videos while loading preferences: "
                + ", ".join(missing_paths[:5])
                + (" ..." if len(missing_paths) > 5 else "")
            )

    def _build_session_download_filename(self, download_type: str, variant: int | None, extension: str) -> str:
        provider = _slugify_filename_part(str(self.video_provider.currentData() or "grok") if hasattr(self, "video_provider") else "grok")
        resolution = _slugify_filename_part(self.video_resolution.currentData() if hasattr(self, "video_resolution") else "auto")
        aspect = _slugify_filename_part(self.video_aspect_ratio.currentData() if hasattr(self, "video_aspect_ratio") else "na")
        item_variant = int(variant or 0)
        session_index = _next_session_download_count()
        normalized_type = _slugify_filename_part(download_type or "download")
        normalized_ext = _slugify_filename_part(extension or "mp4")
        return f"{normalized_type}_{provider}_{resolution}_{aspect}_v{item_variant:02d}_d{session_index:03d}.{normalized_ext}"

    def _thumbnail_for_video(self, video_path: str) -> QIcon:
        source_path = Path(video_path)
        if not source_path.exists():
            return QIcon()

        safe_name = f"thumb_{source_path.stem}_{abs(hash(str(source_path.resolve()))) % 10**10}.jpg"
        thumb_path = THUMBNAILS_DIR / safe_name
        if not thumb_path.exists():
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        "00:00:01.000",
                        "-i",
                        str(source_path),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=144:-2",
                        str(thumb_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception:
                return QIcon()

        pixmap = QPixmap(str(thumb_path))
        if pixmap.isNull():
            return QIcon()
        return QIcon(pixmap)

    def _format_video_date(self, video_path: str) -> str:
        try:
            mtime = Path(video_path).stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "unknown"

    def _selected_video_index(self) -> int:
        if hasattr(self, "video_view_stack") and self.video_view_stack.currentWidget() is self.video_details:
            item = self.video_details.currentItem()
            if item is None:
                return -1
            index = item.data(0, Qt.UserRole)
            return int(index) if isinstance(index, int) else -1
        return self.video_grid.currentRow() if hasattr(self, "video_grid") else -1

    def _set_selected_video_index(self, index: int) -> None:
        if not hasattr(self, "video_grid") or not hasattr(self, "video_details"):
            return

        self.video_grid.blockSignals(True)
        self.video_details.blockSignals(True)

        if 0 <= index < self.video_grid.count():
            self.video_grid.setCurrentRow(index)
        else:
            self.video_grid.clearSelection()
            self.video_grid.setCurrentRow(-1)

        tree_item = self.video_details.topLevelItem(index) if 0 <= index < self.video_details.topLevelItemCount() else None
        self.video_details.setCurrentItem(tree_item)

        self.video_grid.blockSignals(False)
        self.video_details.blockSignals(False)

    def _on_video_details_selection_changed(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None:
            return
        index = current.data(0, Qt.UserRole)
        if isinstance(index, int):
            self._set_selected_video_index(index)
            self.show_selected_video(index)

    def _toggle_video_view_mode(self, details_mode: bool) -> None:
        selected_index = self._selected_video_index()
        self.video_view_stack.setCurrentWidget(self.video_details if details_mode else self.video_grid)
        self.video_view_toggle_btn.setText("☰" if details_mode else "☷")
        self.video_view_toggle_btn.setToolTip(
            "Switch to thumbnail grid view." if details_mode else "Switch to details list view."
        )
        self._set_selected_video_index(selected_index)

    def _refresh_video_picker(self, selected_index: int = -1) -> None:
        self.video_grid.blockSignals(True)
        self.video_details.blockSignals(True)
        self.video_grid.clear()
        self.video_details.clear()

        for index, video in enumerate(self.videos):
            label = self._build_video_label(video)
            video_path = str(video.get("video_file_path") or "")
            icon = self._thumbnail_for_video(video_path)

            grid_item = QListWidgetItem(icon, label)
            grid_item.setToolTip(video_path)
            self.video_grid.addItem(grid_item)

            filename = Path(video_path).name if video_path else "unknown"
            detail_item = QTreeWidgetItem([
                self._format_video_date(video_path),
                filename,
                str(video.get("resolution") or "unknown"),
            ])
            detail_item.setData(0, Qt.UserRole, index)
            detail_item.setToolTip(1, video_path)
            self.video_details.addTopLevelItem(detail_item)

        self.video_grid.blockSignals(False)
        self.video_details.blockSignals(False)

        if not self.videos:
            return

        if selected_index < 0 or selected_index >= len(self.videos):
            selected_index = len(self.videos) - 1
        self._set_selected_video_index(selected_index)
        self.show_selected_video(selected_index)

    def on_video_finished(self, video: dict) -> None:
        self.videos.append(video)
        self._refresh_video_picker(selected_index=len(self.videos) - 1)
        self._append_log(f"Saved: {video['video_file_path']}")

    def open_local_video(self) -> None:
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Video(s)",
            str(self.download_dir),
            "Video Files (*.mp4 *.mov *.mkv *.webm *.avi)",
        )
        if not video_paths:
            return

        for video_path in video_paths:
            resolution = "local"
            try:
                info = self._probe_video_stream_info(Path(video_path))
                resolution = f"{info['width']}x{info['height']}"
            except Exception:
                pass

            loaded_video = {
                "title": f"Opened: {Path(video_path).stem}",
                "prompt": "opened-local-video",
                "resolution": resolution,
                "video_file_path": video_path,
                "source_url": "local-open",
            }
            self.on_video_finished(loaded_video)

    def move_selected_video(self, delta: int) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            return

        target = index + delta
        if target < 0 or target >= len(self.videos):
            return

        self.videos[index], self.videos[target] = self.videos[target], self.videos[index]
        self._refresh_video_picker(selected_index=target)
        self._append_log(f"Reordered videos: moved item to position {target + 1}.")

    def remove_selected_video(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            return

        removed = self.videos.pop(index)
        if not self.videos:
            self._refresh_video_picker(selected_index=-1)
            self.player.stop()
            self.player.setSource(QUrl())
            self.preview_seek_slider.blockSignals(True)
            self.preview_seek_slider.setRange(0, 0)
            self.preview_seek_slider.setValue(0)
            self.preview_seek_slider.blockSignals(False)
            self.preview_position_label.setText("00:00 / 00:00")
        else:
            self._refresh_video_picker(selected_index=min(index, len(self.videos) - 1))

        self._append_log(f"Removed video from list: {removed.get('video_file_path', 'unknown')}")

    def _show_overlay_options_dialog(self) -> tuple[dict[str, object] | None, bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Text Overlay Options")
        layout = QFormLayout(dialog)

        mode_combo = QComboBox(dialog)
        mode_combo.addItem("AI generated from sampled frames", "ai")
        mode_combo.addItem("Manual text", "manual")

        interval_spin = QSpinBox(dialog)
        interval_spin.setRange(1, 60)
        interval_spin.setValue(5)
        interval_spin.setSuffix(" s")

        subtitle_duration_spin = QDoubleSpinBox(dialog)
        subtitle_duration_spin.setRange(0.8, 30.0)
        subtitle_duration_spin.setDecimals(1)
        subtitle_duration_spin.setSingleStep(0.5)
        subtitle_duration_spin.setValue(4.5)
        subtitle_duration_spin.setSuffix(" s")

        text_size_spin = QSpinBox(dialog)
        text_size_spin.setRange(8, 120)
        text_size_spin.setValue(22)

        text_position_combo = QComboBox(dialog)
        text_position_combo.addItem("Bottom", "bottom")
        text_position_combo.addItem("Middle", "middle")
        text_position_combo.addItem("Top", "top")

        font_combo = QFontComboBox(dialog)
        font_combo.setCurrentFont(self.font())

        manual_text = QLineEdit(dialog)
        manual_text.setPlaceholderText("Optional: manual subtitle text, or AI prompt context when in AI mode")
        concept_default = self.concept.toPlainText().strip() if hasattr(self, "concept") else ""
        if concept_default:
            manual_text.setText(concept_default)


        layout.addRow("Mode", mode_combo)
        layout.addRow("Sample interval", interval_spin)
        layout.addRow("Subtitle duration", subtitle_duration_spin)
        layout.addRow("Text size", text_size_spin)
        layout.addRow("Text position", text_position_combo)
        layout.addRow("Font", font_combo)
        layout.addRow("Manual text / AI prompt", manual_text)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None, False

        selected_mode = str(mode_combo.currentData() or "ai")
        manual_value = manual_text.text().strip()
        if selected_mode == "manual" and not manual_value:
            QMessageBox.warning(self, "Manual Text Required", "Enter manual text or choose AI mode.")
            return None, False

        return {
            "mode": selected_mode,
            "interval_seconds": int(interval_spin.value()),
            "subtitle_duration_seconds": float(subtitle_duration_spin.value()),
            "manual_text": manual_value,
            "font_name": font_combo.currentFont().family(),
            "font_size": int(text_size_spin.value()),
            "text_position": str(text_position_combo.currentData() or "bottom"),
        }, True

    def add_overlay_to_selected_video(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video first.")
            return
        if self.overlay_worker is not None and self.overlay_worker.isRunning():
            QMessageBox.information(self, "Overlay In Progress", "Please wait for the current text overlay job to finish.")
            return

        selected_video = self.videos[index]
        input_video = Path(str(selected_video.get("video_file_path") or "")).expanduser()
        if not input_video.exists():
            QMessageBox.warning(self, "Missing Video", "The selected video file no longer exists.")
            return

        options, accepted = self._show_overlay_options_dialog()
        if not accepted or options is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = input_video.with_name(f"{input_video.stem}_overlay_{timestamp}.mp4")

        self.overlay_worker = VideoOverlayWorker(
            input_video=input_video,
            output_video=output_video,
            interval_seconds=int(options["interval_seconds"]),
            subtitle_duration_seconds=float(options["subtitle_duration_seconds"]),
            overlay_mode=str(options["mode"]),
            manual_text=str(options["manual_text"]),
            font_name=str(options.get("font_name") or self.font().family()),
            font_size=int(options.get("font_size") or 22),
            text_position=str(options.get("text_position") or "bottom"),
            ai_callback=self._describe_overlay_frame_with_selected_ai,
            ai_prompt_callback=self._describe_overlay_prompt_with_selected_ai,
            ai_source=str(self.prompt_source.currentData() or "grok"),
        )
        self.overlay_worker.progress.connect(self._append_log)
        self.overlay_worker.finished_overlay.connect(self._on_overlay_finished)
        self.overlay_worker.failed.connect(self._on_overlay_failed)
        self.overlay_worker.start()
        self._append_log(
            f"Started text overlay on '{input_video.name}' (mode={options['mode']}, interval={options['interval_seconds']}s)."
        )

    def _on_overlay_finished(self, video: dict) -> None:
        self.on_video_finished(video)
        self._append_log(f"Text overlay complete: {video.get('video_file_path', '')}")
        QMessageBox.information(self, "Overlay Complete", "Text overlay video created and added to the Generated Videos list.")
        self.overlay_worker = None

    def _on_overlay_failed(self, title: str, message: str) -> None:
        self._append_log(f"ERROR: Text overlay failed: {title}: {message}")
        QMessageBox.critical(self, title, message)
        self.overlay_worker = None

    def _format_time_ms(self, ms: int) -> str:
        total_seconds = max(0, int(ms // 1000))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _on_preview_position_changed(self, position: int) -> None:
        self.preview_seek_slider.blockSignals(True)
        self.preview_seek_slider.setValue(position)
        self.preview_seek_slider.blockSignals(False)
        self.preview_position_label.setText(
            f"{self._format_time_ms(position)} / {self._format_time_ms(self.player.duration())}"
        )
        self._sync_preview_fullscreen_progress()

    def _on_preview_duration_changed(self, duration: int) -> None:
        self.preview_seek_slider.blockSignals(True)
        self.preview_seek_slider.setRange(0, max(0, duration))
        self.preview_seek_slider.blockSignals(False)
        self.preview_position_label.setText(
            f"{self._format_time_ms(self.player.position())} / {self._format_time_ms(duration)}"
        )
        self._sync_preview_fullscreen_progress()

    def seek_preview(self, position: int) -> None:
        self.player.setPosition(max(0, int(position)))

    def _ensure_preview_fullscreen_overlay(self) -> None:
        if self.preview_fullscreen_overlay_btn is not None:
            return

        overlay_btn = QPushButton("🗗")
        overlay_btn.setToolTip("Exit fullscreen preview")
        overlay_btn.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        overlay_btn.setWindowFlag(Qt.FramelessWindowHint, True)
        overlay_btn.clicked.connect(self.toggle_preview_fullscreen)
        self.preview_fullscreen_overlay_btn = overlay_btn

    def _ensure_preview_fullscreen_progress_bar(self) -> None:
        if self.preview_fullscreen_progress_bar is not None:
            return

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 1000)
        progress_bar.setTextVisible(False)
        progress_bar.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        progress_bar.setWindowFlag(Qt.FramelessWindowHint, True)
        progress_bar.setFixedHeight(12)
        progress_bar.setStyleSheet(
            "QProgressBar { background: rgba(10,10,10,0.65); border: 1px solid rgba(220,220,220,0.35); border-radius: 5px; }"
            "QProgressBar::chunk { background-color: #4da3ff; border-radius: 5px; }"
        )
        self.preview_fullscreen_progress_bar = progress_bar

    def _position_preview_fullscreen_overlay(self) -> None:
        if self.preview_fullscreen_overlay_btn is None:
            return

        self.preview_fullscreen_overlay_btn.adjustSize()
        preview_frame = self.preview.frameGeometry()
        x = preview_frame.right() - self.preview_fullscreen_overlay_btn.width() - 24
        y = preview_frame.top() + 20
        self.preview_fullscreen_overlay_btn.move(max(10, x), max(10, y))

    def _position_preview_fullscreen_progress_bar(self) -> None:
        if self.preview_fullscreen_progress_bar is None:
            return

        preview_frame = self.preview.frameGeometry()
        width = max(220, preview_frame.width() - 80)
        x = preview_frame.left() + (preview_frame.width() - width) // 2
        y = preview_frame.bottom() - self.preview_fullscreen_progress_bar.height() - 20
        self.preview_fullscreen_progress_bar.setFixedWidth(width)
        self.preview_fullscreen_progress_bar.move(max(10, x), max(10, y))

    def _sync_preview_fullscreen_progress(self) -> None:
        if self.preview_fullscreen_progress_bar is None:
            return

        duration = max(0, int(self.player.duration()))
        position = max(0, int(self.player.position()))
        if duration <= 0:
            self.preview_fullscreen_progress_bar.setValue(0)
            return
        ratio = min(1.0, position / duration)
        self.preview_fullscreen_progress_bar.setValue(int(ratio * 1000))

    def _on_preview_fullscreen_changed(self, fullscreen: bool) -> None:
        self.preview_fullscreen_btn.setText("🗗" if fullscreen else "⛶")

        if fullscreen:
            self._ensure_preview_fullscreen_overlay()
            self._position_preview_fullscreen_overlay()
            if self.preview_fullscreen_overlay_btn is not None:
                self.preview_fullscreen_overlay_btn.show()
                self.preview_fullscreen_overlay_btn.raise_()

            self._ensure_preview_fullscreen_progress_bar()
            self._position_preview_fullscreen_progress_bar()
            self._sync_preview_fullscreen_progress()
            if self.preview_fullscreen_progress_bar is not None:
                self.preview_fullscreen_progress_bar.show()
                self.preview_fullscreen_progress_bar.raise_()
            self._append_log("Preview entered fullscreen mode.")
            return

        if self.preview_fullscreen_overlay_btn is not None:
            self.preview_fullscreen_overlay_btn.hide()
        if self.preview_fullscreen_progress_bar is not None:
            self.preview_fullscreen_progress_bar.hide()
        self._append_log("Preview exited fullscreen mode.")

    def toggle_preview_fullscreen(self) -> None:
        self.preview.setFullScreen(not self.preview.isFullScreen())

    def on_generation_error(self, error: str) -> None:
        self._append_log(f"ERROR: {error}")
        QMessageBox.critical(self, "Generation Failed", error)

    def show_selected_video(self, index: int) -> None:
        if index < 0 or index >= len(self.videos):
            return
        video = self.videos[index]
        self._preview_video(video["video_file_path"])

    def _preview_video(self, file_path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.player.play()
        self._append_log(f"Selected video for preview: {file_path}")

    def play_preview(self) -> None:
        if self.player.source().isEmpty():
            self._append_log("Preview play requested, but no video is currently loaded.")
            return
        self.player.play()
        self._append_log("Preview playback started.")

    def stop_preview(self) -> None:
        self.player.stop()
        self._append_log("Preview playback stopped.")

    def _set_preview_muted(self, muted: bool) -> None:
        self.preview_muted = bool(muted)
        self.audio_output.setMuted(self.preview_muted)
        self._append_log(f"Preview audio {'muted' if self.preview_muted else 'unmuted'}.")

    def _set_preview_volume(self, value: int) -> None:
        self.preview_volume = int(value)
        self.audio_output.setVolume(self.preview_volume / 100)
        self._append_log(f"Preview volume set to {self.preview_volume}%.")

    def _toggle_prompt_source_fields(self) -> None:
        prompt_source = self.prompt_source.currentData() if hasattr(self, "prompt_source") else "manual"
        video_provider = self.video_provider.currentData() if hasattr(self, "video_provider") else "grok"
        uses_openai = prompt_source == "openai" or video_provider == "openai"
        uses_seedance = video_provider == "seedance"
        uses_grok = prompt_source == "grok" or video_provider == "grok"
        self.manual_prompt.setEnabled(True)
        self.openai_api_key.setEnabled(uses_openai)
        self.openai_access_token.setEnabled(uses_openai)
        self.openai_chat_model.setEnabled(prompt_source == "openai")
        self.seedance_api_key.setEnabled(uses_seedance)
        self.seedance_oauth_token.setEnabled(uses_seedance)
        self.chat_model.setEnabled(uses_grok)
        self.sora_generate_btn.setText("🎬 API Generate Video")
        self.seedance_generate_btn.setText("🎬 API Generate Video")
        self.generate_image_btn.setText("🎬 Create New Video")
        self.generate_image_btn.setEnabled(True)

    def _resolve_download_extension(self, download, download_type: str) -> str:
        suggested = ""
        try:
            suggested = download.downloadFileName() or ""
        except Exception:
            suggested = ""

        if "." in suggested:
            return suggested.rsplit(".", 1)[-1].lower()

        try:
            parsed = urlparse(download.url().toString())
            path_name = Path(parsed.path).name
            if "." in path_name:
                return path_name.rsplit(".", 1)[-1].lower()
        except Exception:
            pass

        return "png" if download_type == "image" else "mp4"

    def _extract_last_frame(self, video_path: str) -> Path | None:
        self._append_log(f"Starting ffmpeg last-frame extraction from: {video_path}")
        extraction_attempts = ["-0", "-0.05", "-0.5", "-1.0"]
        failure_messages: list[str] = []

        for attempt_index, offset in enumerate(extraction_attempts):
            frame_path = self.download_dir / f"last_frame_{int(time.time() * 1000)}_{attempt_index}.png"
            extraction_cmds = _build_last_frame_extraction_cmds(
                video_path,
                ["-sseof", offset],
                frame_path,
            )

            result = None
            for cmd in extraction_cmds:
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    break
                except FileNotFoundError:
                    QMessageBox.critical(self, "ffmpeg Missing", "ffmpeg is required for frame extraction but was not found in PATH.")
                    return None
                except subprocess.CalledProcessError as exc:
                    failure_messages.append(exc.stderr[-500:] or "ffmpeg failed")
                    result = None

            if frame_path.exists() and frame_path.stat().st_size > 0:
                self._append_log(f"Completed ffmpeg last-frame extraction: {frame_path}")
                return frame_path

            if result is not None:
                failure_messages.append(result.stderr[-500:] or "ffmpeg produced an empty frame output")

        QMessageBox.critical(
            self,
            "Frame Extraction Failed",
            "Could not extract the final frame. "
            f"Attempts failed with: {' | '.join(msg.strip() for msg in failure_messages if msg.strip())[:800]}",
        )
        return None

    def _copy_image_to_clipboard(self, frame_path: Path) -> bool:
        self._append_log(f"Copying extracted frame to clipboard: {frame_path}")

        image = QImage(str(frame_path))
        if image.isNull():
            QMessageBox.critical(self, "Frame Extraction Failed", "Frame image could not be loaded.")
            return False

        mime = QMimeData()
        mime.setImageData(image)
        mime.setText(str(frame_path))
        QGuiApplication.clipboard().setMimeData(mime)
        self._append_log("Clipboard image copy completed.")
        return True

    def _upload_frame_into_grok(self, frame_path: Path, on_uploaded=None) -> None:
        import base64

        self._append_log(f"Starting browser-side image paste for frame: {frame_path.name}")

        upload_file_path = frame_path
        upload_file_name = frame_path.name
        suffix = frame_path.suffix.lower().lstrip(".")
        mime_by_ext = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "gif": "image/gif",
        }
        upload_mime = mime_by_ext.get(suffix, "image/png")
        self._append_log(
            f"Continue-from-last-frame: using extracted frame in native format ({upload_file_name}, {upload_mime}) for best color/profile fidelity."
        )

        frame_base64 = base64.b64encode(upload_file_path.read_bytes()).decode("ascii")
        upload_script = r"""
            (() => {
                const base64Data = __FRAME_BASE64__;
                const fileName = __FRAME_NAME__;
                const mimeType = __FRAME_MIME__;
                const selectors = [
                    "textarea[placeholder*='Type to imagine' i]",
                    "input[placeholder*='Type to imagine' i]",
                    "textarea[placeholder*='Type to customize this video' i]",
                    "input[placeholder*='Type to customize this video' i]",
                    "textarea[placeholder*='Type to customize video' i]",
                    "input[placeholder*='Type to customize video' i]",
                    "textarea[placeholder*='Customize video' i]",
                    "input[placeholder*='Customize video' i]",
                    "textarea[aria-label*='Make a video' i]",
                    "input[aria-label*='Make a video' i]",
                    "div.tiptap.ProseMirror[contenteditable='true']",
                    "[contenteditable='true'][aria-label*='Type to imagine' i]",
                    "[contenteditable='true'][data-placeholder*='Type to imagine' i]",
                    "[contenteditable='true'][aria-label*='Type to customize this video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize this video' i]",
                    "[contenteditable='true'][aria-label*='Type to customize video' i]",
                    "[contenteditable='true'][data-placeholder*='Type to customize video' i]",
                    "[contenteditable='true'][aria-label*='Make a video' i]",
                    "[contenteditable='true'][data-placeholder*='Customize video' i]"
                ];
                const isVisible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
                const setInputFiles = (input, files) => {
                    try {
                        const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "files");
                        if (descriptor && typeof descriptor.set === "function") {
                            descriptor.set.call(input, files);
                            return true;
                        }
                    } catch (_) {}

                    try {
                        Object.defineProperty(input, "files", { value: files, configurable: true });
                        return true;
                    } catch (_) {
                        return false;
                    }
                };

                const dispatchFileEvents = (target, dt) => {
                    try {
                        target.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
                        target.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
                    } catch (_) {}

                    ["dragenter", "dragover", "drop"].forEach((eventName) => {
                        try {
                            target.dispatchEvent(new DragEvent(eventName, { bubbles: true, cancelable: true, dataTransfer: dt }));
                        } catch (_) {}
                    });

                    try {
                        const pasteEvent = new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: dt });
                        target.dispatchEvent(pasteEvent);
                    } catch (_) {}
                };

                for (const selector of selectors) {
                    const node = [...document.querySelectorAll(selector)].find((el) => isVisible(el));
                    if (node) {
                        node.focus();
                        const binary = atob(base64Data);
                        const bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
                        const file = new File([bytes], fileName, { type: mimeType });

                        const dt = new DataTransfer();
                        dt.items.add(file);

                        const queryBar = node.closest(".query-bar") || node.closest("form") || node.parentElement;
                        const promptRoot = queryBar?.parentElement || node.parentElement;
                        const scopedInputs = [
                            ...(promptRoot ? promptRoot.querySelectorAll("input[type='file']") : []),
                            ...(queryBar ? queryBar.querySelectorAll("input[type='file']") : []),
                        ];
                        const fileInputs = scopedInputs.length
                            ? [...new Set(scopedInputs)]
                            : [...document.querySelectorAll("input[type='file']")];

                        let populatedInputs = 0;
                        for (const input of fileInputs) {
                            try {
                                if (!setInputFiles(input, dt.files)) continue;
                                dispatchFileEvents(input, dt);
                                populatedInputs += 1;
                            } catch (_) {}
                        }

                        dispatchFileEvents(node, dt);
                        if (queryBar && queryBar !== node) dispatchFileEvents(queryBar, dt);
                        if (promptRoot && promptRoot !== queryBar && promptRoot !== node) dispatchFileEvents(promptRoot, dt);

                        return {
                            ok: populatedInputs > 0,
                            fileInputs: fileInputs.length,
                            populatedInputs,
                            selector,
                        };
                    }
                }
                return { ok: false, error: 'Prompt input not found for paste' };
            })()
        """

        upload_script = (
            upload_script.replace("__FRAME_BASE64__", repr(frame_base64))
            .replace("__FRAME_NAME__", repr(upload_file_name))
            .replace("__FRAME_MIME__", repr(upload_mime))
        )

        def after_focus(_result):
            if callable(on_uploaded):
                on_uploaded()

        self.browser.page().runJavaScript(upload_script, after_focus)

    def _wait_for_continue_upload_reload(self) -> None:
        self.continue_from_frame_waiting_for_reload = True
        self.continue_from_frame_reload_timeout_timer.start(10000)
        self._append_log(
            "Continue-from-last-frame: image pasted. Grok should auto-reload after upload; "
            "waiting for the new page before entering the continuation prompt..."
        )

    def _on_continue_reload_timeout(self) -> None:
        if not self.continue_from_frame_waiting_for_reload or not self.continue_from_frame_active:
            return
        self.continue_from_frame_waiting_for_reload = False
        self._append_log(
            "Timed out waiting for upload-triggered reload; continuing with prompt submission."
        )
        self._start_manual_browser_generation(self.continue_from_frame_prompt, 1)

    def continue_from_last_frame(self) -> None:
        latest_video = self._resolve_latest_video_for_continuation()
        if not latest_video:
            QMessageBox.warning(self, "No Videos", "Generate or open a video first.")
            return

        manual_prompt = self.manual_prompt.toPlainText().strip()
        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Enter a manual prompt for the continuation run.")
            return

        self.continue_from_frame_active = True
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer.stop()
        self.continue_from_frame_target_count = self.count.value()
        self._start_automation_counter_tracking(self.continue_from_frame_target_count)
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = manual_prompt
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = None
        self._append_log(
            f"Continue-from-last-frame started for {self.continue_from_frame_target_count} iteration(s)."
        )
        self._append_log(f"Continue-from-last-frame source video selected: {latest_video}")
        self._start_continue_iteration()

    def continue_from_local_image(self) -> None:
        manual_prompt = self.manual_prompt.toPlainText().strip()
        if not manual_prompt:
            QMessageBox.warning(self, "Missing Manual Prompt", "Enter a manual prompt for the continuation run.")
            return

        image_path, _ = QFileDialog.getOpenFileName(self, "Select image", str(self.download_dir), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not image_path:
            return

        seed_image = Path(image_path)
        if not seed_image.exists():
            QMessageBox.warning(self, "Image Missing", "Selected image was not found on disk.")
            return

        self.continue_from_frame_active = True
        self.continue_from_frame_waiting_for_reload = False
        self.continue_from_frame_reload_timeout_timer.stop()
        self.continue_from_frame_target_count = self.count.value()
        self._start_automation_counter_tracking(self.continue_from_frame_target_count)
        self.continue_from_frame_completed = 0
        self.continue_from_frame_prompt = manual_prompt
        self.continue_from_frame_current_source_video = ""
        self.continue_from_frame_seed_image_path = seed_image

        self._append_log(
            f"Continue-from-image started for {self.continue_from_frame_target_count} iteration(s) using {seed_image}."
        )
        self._start_continue_iteration()

    def _training_output_dir(self) -> Path:
        output_dir = Path(self.training_output_dir.text().strip() or str(self.download_dir / "browser_training"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.training_output_dir.setText(str(output_dir))
        return output_dir

    def _choose_training_output_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose Training Output Folder", self.training_output_dir.text().strip() or str(self.download_dir))
        if selected:
            self.training_output_dir.setText(selected)

    def _choose_training_trace_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Trace",
            self.training_output_dir.text().strip() or str(self.download_dir),
            "Trace JSON (raw_training_trace.json *.json)",
        )
        if file_path:
            self.training_trace_path.setText(file_path)

    def _choose_training_process_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Process",
            self.training_output_dir.text().strip() or str(self.download_dir),
            "Process JSON (*.process.json *.json)",
        )
        if file_path:
            self.training_process_path.setText(file_path)

    def _set_training_button_states(self) -> None:
        self.training_start_btn.setEnabled(not self.embedded_training_active)
        self.training_stop_btn.setEnabled(self.embedded_training_active)

    def _inject_embedded_training_capture_script(self) -> None:
        script = r"""
            (() => {
              const createCssPath = (el) => {
                if (!el || !(el instanceof Element)) return '';
                const parts = [];
                let node = el;
                while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 6) {
                  let part = node.nodeName.toLowerCase();
                  if (node.id) {
                    part += `#${node.id}`;
                    parts.unshift(part);
                    break;
                  }
                  const classes = (node.className || '').toString().trim().split(/\s+/).filter(Boolean).slice(0,2).join('.');
                  if (classes) part += `.${classes}`;
                  const parent = node.parentElement;
                  if (parent) {
                    const siblings = Array.from(parent.children).filter((x) => x.nodeName === node.nodeName);
                    if (siblings.length > 1) {
                      part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
                    }
                  }
                  parts.unshift(part);
                  node = node.parentElement;
                }
                return parts.join(' > ');
              };

              const scrub = (value) => (value || '').toString().slice(0, 400);

              if (window.__grokTrainer && typeof window.__grokTrainer.cleanup === 'function') {
                window.__grokTrainer.cleanup();
              }

              const state = { events: [] };
              const push = (event) => {
                state.events.push({ ...event, url: window.location.href });
              };

              const onClick = (event) => {
                const el = event.target;
                push({
                  action: 'click',
                  selector: createCssPath(el),
                  text: scrub(el && el.innerText),
                  tag: (el && el.tagName || '').toLowerCase(),
                });
              };

              const onChange = (event) => {
                const el = event.target;
                if (!el) return;
                const tag = (el.tagName || '').toLowerCase();
                if (!['input', 'textarea', 'select'].includes(tag)) return;
                push({
                  action: 'fill',
                  selector: createCssPath(el),
                  value: scrub(el.value),
                  tag,
                });
              };

              const onKeyDown = (event) => {
                if (!event.key || !['Enter', 'Tab', 'Escape'].includes(event.key)) return;
                const el = event.target;
                push({
                  action: 'press',
                  selector: createCssPath(el),
                  value: event.key,
                  tag: (el && el.tagName || '').toLowerCase(),
                });
              };

              document.addEventListener('click', onClick, true);
              document.addEventListener('change', onChange, true);
              document.addEventListener('keydown', onKeyDown, true);

              state.pull = () => {
                const events = state.events.slice();
                state.events = [];
                return events;
              };

              state.cleanup = () => {
                document.removeEventListener('click', onClick, true);
                document.removeEventListener('change', onChange, true);
                document.removeEventListener('keydown', onKeyDown, true);
              };

              window.__grokTrainer = state;
              return true;
            })();
        """
        self.browser.page().runJavaScript(script)

    def _poll_embedded_training_events(self) -> None:
        if not self.embedded_training_active:
            return

        script = "(() => { if (!window.__grokTrainer || !window.__grokTrainer.pull) return []; return window.__grokTrainer.pull(); })();"

        def _after_poll(result):
            if not self.embedded_training_active:
                return
            if not isinstance(result, list):
                return
            for item in result:
                if isinstance(item, dict):
                    self._record_embedded_training_event(item)

        self.browser.page().runJavaScript(script, _after_poll)

    def _record_embedded_training_event(self, event: dict) -> None:
        if not self.embedded_training_active or self.embedded_training_output_dir is None:
            return

        self.embedded_training_event_counter += 1
        event_record = {
            "index": self.embedded_training_event_counter,
            "source": "embedded-browser",
            "ts": round(time.time() - self.embedded_training_started_at, 3),
            **event,
        }

        screenshots_dir = self.embedded_training_output_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_name = f"step-{self.embedded_training_event_counter:03d}.png"
        screenshot_path = screenshots_dir / screenshot_name
        pixmap = self.browser.grab()
        if not pixmap.isNull() and pixmap.save(str(screenshot_path)):
            event_record["screenshot"] = str(Path("screenshots") / screenshot_name)
        else:
            event_record["screenshot"] = ""

        self.embedded_training_events.append(event_record)

    def _start_browser_training_worker(self, mode: str, payload: dict) -> None:
        if self.browser_training_worker is not None and self.browser_training_worker.isRunning():
            QMessageBox.information(self, "Training in Progress", "Please wait until the current training task finishes.")
            return

        self.browser_training_worker = BrowserTrainingWorker(mode, payload)
        self.browser_training_worker.status.connect(self._on_browser_training_status)
        self.browser_training_worker.finished.connect(self._on_browser_training_finished)
        self.browser_training_worker.failed.connect(self._on_browser_training_failed)
        self.browser_training_worker.start()
        self.training_status.setText(f"Status: {mode} started")

    def _on_browser_training_status(self, message: str) -> None:
        self.training_status.setText(f"Status: {message}")
        self._append_log(message)

    def _on_browser_training_finished(self, mode: str, output_path: str) -> None:
        self.training_status.setText(f"Status: {mode} complete")
        if mode == "train":
            self.training_trace_path.setText(output_path)
            self._append_log(f"Training complete. Trace saved: {output_path}")
        elif mode == "build":
            self.training_process_path.setText(output_path)
            self._append_log(f"Process build complete. Process saved: {output_path}")
        elif mode == "run":
            self._append_log(f"Process run complete. Report saved: {output_path}")

        QMessageBox.information(self, "AI Flow Trainer", f"{mode.title()} completed.\n{output_path}")

    def _on_browser_training_failed(self, error_message: str) -> None:
        self.training_status.setText("Status: failed")
        self._append_log(f"ERROR: Browser training workflow failed: {error_message}")
        QMessageBox.critical(self, "AI Flow Trainer", error_message)

    def start_browser_training(self) -> None:
        if self.embedded_training_active:
            QMessageBox.information(self, "Training In Progress", "Embedded training is already running. Use Stop Training first.")
            return

        if self.training_use_embedded_browser.isChecked():
            output_dir = self._training_output_dir()
            self.embedded_training_output_dir = output_dir
            self.embedded_training_events = []
            self.embedded_training_started_at = time.time()
            self.embedded_training_event_counter = 0
            self.embedded_training_active = True
            self._set_training_button_states()

            start_url = self.training_start_url.text().strip() or "https://grok.com/imagine"
            self.browser_tabs.setCurrentIndex(self.grok_browser_tab_index)
            current_url = self.browser.url().toString().strip()
            if current_url != start_url:
                self.browser.setUrl(QUrl(start_url))
            else:
                self._inject_embedded_training_capture_script()

            self.embedded_training_poll_timer.start()
            self.training_status.setText("Status: embedded training started")
            self._append_log("Embedded browser training started. Interact in the Browser tab, then click Stop Training.")
            return

        output_dir = self._training_output_dir()
        start_url = self.training_start_url.text().strip() or "https://grok.com/imagine"
        timeout_s = self.training_timeout.value()
        self._start_browser_training_worker(
            "train",
            {
                "start_url": start_url,
                "output_dir": output_dir,
                "timeout_s": timeout_s,
            },
        )

    def stop_browser_training(self) -> None:
        if not self.embedded_training_active:
            QMessageBox.information(self, "No Active Training", "Embedded training is not currently running.")
            return

        self.embedded_training_poll_timer.stop()
        self.browser.page().runJavaScript(
            "(() => { if (window.__grokTrainer && window.__grokTrainer.cleanup) { window.__grokTrainer.cleanup(); } return true; })();"
        )

        output_dir = self.embedded_training_output_dir or self._training_output_dir()
        trace = {
            "start_url": self.training_start_url.text().strip() or "https://grok.com/imagine",
            "browser": "qtwebengine-embedded",
            "created_at": int(time.time()),
            "events": self.embedded_training_events,
        }
        trace_path = output_dir / "raw_training_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

        self.embedded_training_active = False
        self.embedded_training_output_dir = None
        self._set_training_button_states()
        self.training_trace_path.setText(str(trace_path))
        self.training_status.setText("Status: embedded training complete")
        self._append_log(f"Embedded training stopped. Trace saved: {trace_path}")
        QMessageBox.information(self, "AI Flow Trainer", f"Training completed.\n{trace_path}")

    def build_browser_training_process(self) -> None:
        trace_path_text = self.training_trace_path.text().strip()
        if not trace_path_text:
            QMessageBox.warning(self, "Missing Trace", "Select or generate a training trace before building a process.")
            return

        credential = self.openai_api_key.text().strip() or self.openai_access_token.text().strip()
        if not credential:
            QMessageBox.warning(self, "Missing OpenAI Credentials", "Enter an OpenAI API key or sign in with Browser Authorization in Settings.")
            return

        trace_path = Path(trace_path_text)
        if not trace_path.exists():
            QMessageBox.warning(self, "Trace Not Found", f"Training trace does not exist:\n{trace_path}")
            return

        model = self.training_openai_model.text().strip() or "gpt-5.1-codex"
        self._start_browser_training_worker(
            "build",
            {
                "trace_path": trace_path,
                "access_token": credential,
                "model": model,
            },
        )

    def run_browser_training_process(self) -> None:
        process_path_text = self.training_process_path.text().strip()
        if not process_path_text:
            QMessageBox.warning(self, "Missing Process", "Select or build a process file before running it.")
            return

        process_path = Path(process_path_text)
        if not process_path.exists():
            QMessageBox.warning(self, "Process Not Found", f"Process file does not exist:\n{process_path}")
            return

        output_dir = self._training_output_dir() / "run_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        start_url = self.training_start_url.text().strip() or "https://grok.com/imagine"

        self._start_browser_training_worker(
            "run",
            {
                "process_path": process_path,
                "start_url": start_url,
                "output_dir": output_dir,
                "timeout_s": 180,
            },
        )

    def _on_browser_tab_changed(self, index: int) -> None:
        if index == getattr(self, "grok_browser_tab_index", -1):
            self.browser = self.grok_browser_view
        elif index == getattr(self, "sora_browser_tab_index", -1):
            self.browser = self.sora_browser

    def show_browser_page(self) -> None:
        if not self._is_browser_tab_enabled("Grok"):
            self._append_log("Grok Browser tab is disabled. Re-enable it from View → Browser Tabs.")
            return
        self.browser_tabs.setCurrentIndex(self.grok_browser_tab_index)
        self.browser = self.grok_browser_view
        self.grok_browser_view.setUrl(QUrl("https://grok.com/imagine"))
        self._append_log("Navigated embedded browser to grok.com/imagine.")

    def show_sora_browser_page(self) -> None:
        if not self._is_browser_tab_enabled("Sora"):
            self._append_log("Sora Browser tab is disabled. Re-enable it from View → Browser Tabs.")
            return
        self.browser_tabs.setCurrentIndex(self.sora_browser_tab_index)
        self.browser = self.sora_browser
        self.sora_browser.setUrl(QUrl("https://sora.chatgpt.com/drafts"))
        self._append_log("Navigated embedded browser to sora.chatgpt.com/drafts.")

    def stitch_all_videos(self) -> None:
        if self.stitch_worker and self.stitch_worker.isRunning():
            QMessageBox.information(self, "Stitch In Progress", "A stitch operation is already running.")
            return

        if len(self.videos) < 2:
            QMessageBox.warning(self, "Need More Videos", "At least two videos are required to stitch.")
            return

        timestamp = int(time.time() * 1000)
        output_file = self.download_dir / f"stitched_{timestamp}.mp4"
        stitched_base_file = self.download_dir / f"stitched_base_{timestamp}.mp4"
        video_paths = [Path(video["video_file_path"]) for video in self.videos]
        interpolate_enabled = self.stitch_interpolation_checkbox.isChecked()
        interpolation_fps = int(self.stitch_interpolation_fps.currentData())
        upscale_enabled = self.stitch_upscale_checkbox.isChecked()
        upscale_target = str(self.stitch_upscale_target.currentData())
        crossfade_enabled = self.stitch_crossfade_checkbox.isChecked()
        gpu_requested = self.stitch_gpu_checkbox.isChecked()
        gpu_enabled = gpu_requested and self._ffmpeg_supports_nvenc()
        custom_music_file = self.custom_music_file
        mute_original_audio = self.stitch_mute_original_checkbox.isChecked()
        original_audio_volume = float(self.stitch_original_audio_volume.value()) / 100.0
        music_volume = float(self.stitch_music_volume.value()) / 100.0
        audio_fade_duration = self.stitch_audio_fade_duration.value()
        if gpu_requested and not gpu_enabled:
            self._append_log("GPU encoding requested, but ffmpeg NVENC is unavailable. Falling back to CPU encoding.")

        settings_summary = (
            f"Crossfade: {'on' if crossfade_enabled else 'off'}"
            + (f" ({self.crossfade_duration.value():.1f}s)" if crossfade_enabled else "")
            + f" | Interpolation: {f'{interpolation_fps} fps' if interpolate_enabled else 'off'}"
            + f" | Upscaling: {upscale_target if upscale_enabled else 'off'}"
            + f" | Encode: {'GPU' if gpu_enabled else 'CPU'}"
            + (f" | Music: {custom_music_file.name}" if custom_music_file else " | Music: off")
        )

        started_at = time.time()
        self.stitch_progress_label.setVisible(True)
        self.stitch_progress_bar.setVisible(True)
        self.statusBar().show()
        self._refresh_status_bar_visibility()

        def update_progress(value: int, stage: str) -> None:
            bounded_value = max(0, min(100, int(value)))
            elapsed = time.time() - started_at
            eta_label = "calculating..."
            if bounded_value > 0:
                eta_seconds = max(0.0, (elapsed / bounded_value) * (100 - bounded_value))
                eta_label = f"~{eta_seconds:.0f}s"

            self.stitch_progress_bar.setValue(bounded_value)
            self.stitch_progress_label.setText(
                f"Stitch progress: {stage} | {bounded_value}% | Elapsed: {elapsed:.1f}s | ETA: {eta_label} | {settings_summary}"
            )

        def on_stitch_failed(title: str, message: str) -> None:
            self.stitch_progress_label.setText(f"Stitch progress: failed ({message[:120]})")
            self.stitch_progress_bar.setVisible(False)
            self.stitch_progress_label.setVisible(False)
            self.statusBar().hide()
            self._refresh_status_bar_visibility()
            QMessageBox.critical(self, title, message)

        def on_stitch_finished(stitched_video: dict) -> None:
            self._append_log(f"Stitched video created: {stitched_video['video_file_path']}")
            self.stitch_progress_label.setText("Stitch progress: complete")
            self.stitch_progress_bar.setValue(100)
            self.stitch_progress_bar.setVisible(False)
            self.stitch_progress_label.setVisible(False)
            self.statusBar().hide()
            self._refresh_status_bar_visibility()
            self.on_video_finished(stitched_video)

        def on_stitch_complete() -> None:
            self.stitch_worker = None

        update_progress(1, "Preparing stitch pipeline...")

        self.stitch_worker = StitchWorker(
            window=self,
            video_paths=video_paths,
            output_file=output_file,
            stitched_base_file=stitched_base_file,
            crossfade_enabled=crossfade_enabled,
            crossfade_duration=self.crossfade_duration.value(),
            interpolate_enabled=interpolate_enabled,
            interpolation_fps=interpolation_fps,
            upscale_enabled=upscale_enabled,
            upscale_target=upscale_target,
            use_gpu_encoding=gpu_enabled,
            custom_music_file=custom_music_file,
            mute_original_audio=mute_original_audio,
            original_audio_volume=original_audio_volume,
            music_volume=music_volume,
            audio_fade_duration=audio_fade_duration,
        )
        self.stitch_worker.progress.connect(update_progress)
        self.stitch_worker.status.connect(self._append_log)
        self.stitch_worker.failed.connect(on_stitch_failed)
        self.stitch_worker.finished_stitch.connect(on_stitch_finished)
        self.stitch_worker.finished.connect(on_stitch_complete)
        self.stitch_worker.start()

    def _stitch_videos_concat(
        self,
        video_paths: list[Path],
        output_file: Path,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        list_file = self.download_dir / f"stitch_list_{int(time.time() * 1000)}.txt"

        concat_lines = []
        for video_path in video_paths:
            quoted_path = video_path.as_posix().replace("'", "'\\''")
            concat_lines.append(f"file '{quoted_path}'")
        list_file.write_text("\n".join(concat_lines), encoding="utf-8")

        try:
            total_duration = sum(self._probe_video_duration(path) for path in video_paths)
            self._run_ffmpeg_with_progress(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    *self._video_encoder_args(use_gpu_encoding),
                    *self._mp4_compatibility_args(),
                    "-c:a",
                    "aac",
                    str(output_file),
                ],
                total_duration=total_duration,
                progress_callback=progress_callback,
            )
        finally:
            if list_file.exists():
                list_file.unlink()

    def _probe_video_duration(self, video_path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise RuntimeError(f"Could not determine valid duration for {video_path.name}.")
        return duration

    def _probe_video_stream_info(self, video_path: Path) -> dict:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(f"No video stream found in {video_path.name}.")
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration_raw = stream.get("duration")
        duration = float(duration_raw) if duration_raw not in (None, "N/A", "") else self._probe_video_duration(video_path)
        if width <= 0 or height <= 0 or duration <= 0:
            raise RuntimeError(f"Could not probe valid video stream info for {video_path.name}.")
        return {"width": width, "height": height, "duration": duration}

    def _video_has_audio_stream(self, video_path: Path) -> bool:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return bool(result.stdout.strip())

    def _ffmpeg_supports_nvenc(self) -> bool:
        if self._ffmpeg_nvenc_checked:
            return self._ffmpeg_nvenc_available

        self._ffmpeg_nvenc_checked = True
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            encoders_output = f"{result.stdout}\n{result.stderr}"
            self._ffmpeg_nvenc_available = "h264_nvenc" in encoders_output
        except Exception:
            self._ffmpeg_nvenc_available = False
        self.preview_fullscreen_overlay_btn: QPushButton | None = None

        return self._ffmpeg_nvenc_available

    def _video_encoder_args(self, use_gpu_encoding: bool, crf: int = 20) -> list[str]:
        if use_gpu_encoding and self._ffmpeg_supports_nvenc():
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(max(18, min(30, crf + 1))), "-b:v", "0"]
        return ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf)]

    def _mp4_compatibility_args(self) -> list[str]:
        # Keep stitched output broadly compatible with Windows thumbnails, Android playback,
        # and social upload transcoders.
        return ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    def _run_ffmpeg_with_progress(
        self,
        ffmpeg_cmd: list[str],
        total_duration: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        command = ffmpeg_cmd[:-1] + ["-progress", "pipe:1", "-nostats", "-loglevel", "error", ffmpeg_cmd[-1]]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._active_ffmpeg_process = process

        if progress_callback is not None:
            progress_callback(0.0)

        out_time_ms = 0
        output_lines: list[str] = []
        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    if self.stop_all_requested:
                        process.terminate()
                    line = raw_line.strip()
                    if not line:
                        continue
                    output_lines.append(line)
                    if len(output_lines) > 200:
                        output_lines = output_lines[-200:]
                    if line.startswith("out_time_ms="):
                        try:
                            out_time_ms = int(line.split("=", 1)[1])
                        except ValueError:
                            continue
                        if total_duration > 0 and progress_callback is not None:
                            progress = (out_time_ms / 1_000_000.0) / total_duration
                            progress_callback(max(0.0, min(1.0, progress)))

            return_code = process.wait()
            if self.stop_all_requested:
                stderr_text = "\n".join(output_lines[-80:])
                raise RuntimeError(f"Stitch/encode stopped by user. {stderr_text}".strip())
            if return_code != 0:
                stderr_text = "\n".join(output_lines[-80:])
                raise subprocess.CalledProcessError(return_code, command, stderr=stderr_text)

            if progress_callback is not None:
                progress_callback(1.0)
        finally:
            self._active_ffmpeg_process = None

    def _stitch_videos_with_crossfade(
        self,
        video_paths: list[Path],
        output_file: Path,
        crossfade_duration: float,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        stream_infos = [self._probe_video_stream_info(path) for path in video_paths]
        durations = [info["duration"] for info in stream_infos]
        has_audio = all(self._video_has_audio_stream(path) for path in video_paths)
        for path, duration in zip(video_paths, durations):
            if duration <= crossfade_duration + 0.05:
                raise RuntimeError(
                    f"Clip '{path.name}' is too short ({duration:.2f}s). Each clip must be longer than {crossfade_duration:.1f}s for crossfade stitching."
                )

        target_width = stream_infos[0]["width"]
        target_height = stream_infos[0]["height"]

        ffmpeg_cmd = ["ffmpeg", "-y"]
        for path in video_paths:
            ffmpeg_cmd.extend(["-i", str(path)])

        filter_parts: list[str] = []
        for idx in range(len(video_paths)):
            filter_parts.append(
                f"[{idx}:v]fps=24,scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1,settb=AVTB,format=yuv420p[vsrc{idx}]"
            )
            if has_audio:
                filter_parts.append(
                    f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,aresample=async=1[asrc{idx}]"
                )

        cumulative_duration = durations[0]
        video_prev = "vsrc0"
        audio_prev = "asrc0"

        for idx in range(1, len(video_paths)):
            offset = cumulative_duration - crossfade_duration
            next_video = f"v{idx}"
            filter_parts.append(
                f"[{video_prev}][vsrc{idx}]xfade=transition=fade:duration={crossfade_duration:.3f}:offset={max(0.0, offset):.3f}[{next_video}]"
            )
            video_prev = next_video

            if has_audio:
                next_audio = f"a{idx}"
                filter_parts.append(
                    f"[{audio_prev}][asrc{idx}]acrossfade=d={crossfade_duration:.3f}:c1=tri:c2=tri[{next_audio}]"
                )
                audio_prev = next_audio

            cumulative_duration += durations[idx] - crossfade_duration

        filter_complex = ";".join(filter_parts)
        ffmpeg_cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                f"[{video_prev}]",
                *self._video_encoder_args(use_gpu_encoding),
                *self._mp4_compatibility_args(),
                str(output_file),
            ]
        )
        if has_audio:
            ffmpeg_cmd[-1:-1] = ["-map", f"[{audio_prev}]", "-c:a", "aac"]

        self._run_ffmpeg_with_progress(
            ffmpeg_cmd,
            total_duration=cumulative_duration,
            progress_callback=progress_callback,
        )

    def _enhance_stitched_video(
        self,
        input_file: Path,
        output_file: Path,
        interpolate: bool,
        interpolation_fps: int,
        upscale: bool,
        upscale_target: str = "2x",
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        if not interpolate and not upscale:
            return

        vf_filters: list[str] = []
        if interpolate:
            target_fps = 48 if interpolation_fps not in {48, 60} else interpolation_fps
            vf_filters.append(
                f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
            )
        if upscale:
            if upscale_target == "1080p":
                vf_filters.append(
                    "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                )
            elif upscale_target == "1440p":
                vf_filters.append(
                    "scale=2560:1440:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=2560:1440:(ow-iw)/2:(oh-ih)/2"
                )
            elif upscale_target == "4k":
                vf_filters.append(
                    "scale=3840:2160:force_original_aspect_ratio=decrease:flags=lanczos,"
                    "pad=3840:2160:(ow-iw)/2:(oh-ih)/2"
                )
            else:
                vf_filters.append("scale='min(iw*2,3840)':'min(ih*2,2160)':flags=lanczos")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-vf",
            ",".join(vf_filters),
            *self._video_encoder_args(use_gpu_encoding, crf=18),
            *self._mp4_compatibility_args(),
            "-c:a",
            "copy",
            str(output_file),
        ]
        self._run_ffmpeg_with_progress(
            command,
            total_duration=self._probe_video_duration(input_file),
            progress_callback=progress_callback,
        )

    def _apply_custom_music_track(
        self,
        input_file: Path,
        output_file: Path,
        music_file: Path,
        mute_original_audio: bool,
        original_audio_volume: float,
        music_volume: float,
        fade_duration: float,
        progress_callback: Callable[[float], None] | None = None,
        use_gpu_encoding: bool = False,
    ) -> None:
        temp_output = self.download_dir / f"stitch_music_{int(time.time() * 1000)}.mp4"
        video_duration = self._probe_video_duration(input_file)
        music_duration = self._probe_video_duration(music_file)

        clamped_original_volume = max(0.0, min(2.0, original_audio_volume))
        clamped_music_volume = max(0.0, min(2.0, music_volume))
        clamped_fade = max(0.0, min(float(fade_duration), max(0.0, video_duration / 2.0)))

        trim_start = 0.0
        trim_duration = video_duration
        if music_duration > video_duration:
            extra = music_duration - video_duration
            trim_start = extra / 2.0
            trim_duration = video_duration

        has_original_audio = self._video_has_audio_stream(input_file)
        music_chain = (
            f"[1:a]atrim=start={trim_start:.6f}:duration={trim_duration:.6f},asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            f"volume={clamped_music_volume:.3f}[music]"
        )

        filter_parts: list[str] = [music_chain]
        audio_output_label = "music"

        if has_original_audio and not mute_original_audio:
            filter_parts.append(
                f"[0:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"volume={clamped_original_volume:.3f}[orig]"
            )
            filter_parts.append("[orig][music]amix=inputs=2:duration=first:dropout_transition=0[audmix]")
            audio_output_label = "audmix"

        if clamped_fade > 0.0:
            fade_out_start = max(0.0, video_duration - clamped_fade)
            filter_parts.append(
                f"[{audio_output_label}]afade=t=in:st=0:d={clamped_fade:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={clamped_fade:.3f}[audfinal]"
            )
            audio_output_label = "audfinal"

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-i",
            str(music_file),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v",
            "-map",
            f"[{audio_output_label}]",
            *self._video_encoder_args(use_gpu_encoding),
            *self._mp4_compatibility_args(),
            "-c:a",
            "aac",
            "-shortest",
            str(temp_output),
        ]

        self._run_ffmpeg_with_progress(
            command,
            total_duration=video_duration,
            progress_callback=progress_callback,
        )

        temp_output.replace(output_file)

    def upload_selected_to_youtube(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        title, description, hashtags, category_id, accepted = self._show_upload_dialog("YouTube")
        if not accepted:
            return

        client_secret_path = BASE_DIR / "client_secret.json"
        client_secret_file = str(client_secret_path) if client_secret_path.exists() else ""
        token_file = str(BASE_DIR / "youtube_token.json")
        if not client_secret_file and not Path(token_file).exists():
            QMessageBox.critical(
                self,
                "Missing YouTube OAuth credentials",
                "No youtube_token.json found and client_secret.json is missing. "
                "Provide one of them to upload via YouTube OAuth.",
            )
            return

        self._start_upload(
            platform_name="YouTube",
            upload_fn=upload_video_to_youtube,
            upload_kwargs={
                "client_secret_file": client_secret_file,
                "token_file": token_file,
                "video_path": video_path,
                "title": title,
                "description": description,
                "tags": hashtags,
                "category_id": category_id,
                "youtube_api_key": self.youtube_api_key.text().strip(),
            },
            success_dialog_title="YouTube Upload Complete",
            success_prefix="Video uploaded successfully. ID:",
        )

    def _upload_video_to_facebook_page(self, page_id, access_token, video_path, title, description):
        video_size = os.path.getsize(video_path)
        version = "v24.0"  # or latest stable, check developers.facebook.com/docs/graph-api/changelog

        headers = {"Authorization": f"Bearer {access_token}"}  # sometimes not needed if in params

        # Step 1: Start upload session
        start_url = f"https://graph-video.facebook.com/{version}/{page_id}/videos"
        start_params = {
            "access_token": access_token,
            "upload_phase": "start",
            "file_size": video_size,
            "title": title,
            "description": description,
            "published": False,
        }
        start_resp = requests.post(start_url, params=start_params)
        #start_resp.raise_for_status()
        start_data = start_resp.json()
        
        print("DEBUG Start response:", start_data)

        if 'video_id' not in start_data or 'upload_url' not in start_data:
            QMessageBox.warning(self, "Facebook Upload","Start failed, no video_id/upload_url: {start_data}")
            return
        

        video_id = start_data['video_id']
        upload_session_id = start_data.get('upload_session_id', video_id)
        upload_url = start_data['upload_url']  # or construct if needed

        # Step 2: Upload the file (single chunk for small videos; chunk for large)
        with open(video_path, "rb") as f:
            upload_headers = {"Content-Type": "application/octet-stream"}
            upload_resp = requests.post(upload_url, headers=upload_headers, data=f)
            upload_resp.raise_for_status()
            upload_data = upload_resp.json()

        print("DEBUG Upload response:", upload_data)

        if not upload_data.get('success'):
            raise ValueError(f"Upload failed: {upload_data}")

        # Step 3: Finish / publish (sometimes automatic, sometimes explicit)
        # If needed: POST again with upload_phase=finish, upload_session_id, etc.

        # ── Step 3: Finish / publish ──
        finish_url = f"https://graph-video.facebook.com/{version}/{page_id}/videos"
        finish_params = {
            "access_token": access_token,
            "upload_phase": "finish",
            "upload_session_id": upload_session_id,
            "title": title,
            "description": description,
            "published": False,                     # ← key: do NOT publish
        }
        return video_id  # This is what your worker expects

    def start_facebook_browser_upload(self) -> None:
        self._refresh_facebook_upload_tab_url()
        video_path = self._selected_video_path_for_upload()
        if not video_path:
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return
        if not Path(video_path).exists():
            QMessageBox.warning(self, "Missing Video", "The selected generated video file no longer exists on disk.")
            return

        title, description, hashtags, _, accepted = self._show_upload_dialog("Facebook")
        if not accepted:
            return

        caption_text = self._compose_social_text(description, hashtags)
        if not caption_text:
            caption_text = self._compose_social_text(self.ai_social_metadata.description, self.ai_social_metadata.hashtags)

        self._run_social_upload_via_mode(
            platform_name="Facebook",
            video_path=video_path,
            caption=caption_text,
            title=title,
        )

    def start_instagram_browser_upload(self) -> None:
        video_path = self._selected_video_path_for_upload()
        if not video_path:
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return
        if not Path(video_path).exists():
            QMessageBox.warning(self, "Missing Video", "The selected generated video file no longer exists on disk.")
            return

        _, caption, hashtags, _, accepted = self._show_upload_dialog("Instagram", title_enabled=False)
        if not accepted:
            return

        self._run_social_upload_via_mode(
            platform_name="Instagram",
            video_path=video_path,
            caption=self._compose_social_text(caption, hashtags),
            title="",
        )

    def start_tiktok_browser_upload(self) -> None:
        video_path = self._selected_video_path_for_upload()
        if not video_path:
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return
        if not Path(video_path).exists():
            QMessageBox.warning(self, "Missing Video", "The selected generated video file no longer exists on disk.")
            return

        title, caption, hashtags, _, accepted = self._show_upload_dialog("TikTok", title_enabled=True)
        if not accepted:
            return

        filename_title = title.strip() or self.ai_social_metadata.medium_title.strip() or caption.strip()
        filename_slogan = self.ai_social_metadata.tiktok_subheading.strip()
        renamed_video_path = self._stage_tiktok_browser_video(video_path, filename_title, filename_slogan, hashtags)

        self._run_social_upload_via_mode(
            platform_name="TikTok",
            video_path=renamed_video_path,
            caption=self._compose_social_text(caption, hashtags),
            title=title,
        )


    def start_youtube_browser_upload(self) -> None:
        video_path = self._selected_video_path_for_upload()
        if not video_path:
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return
        if not Path(video_path).exists():
            QMessageBox.warning(self, "Missing Video", "The selected generated video file no longer exists on disk.")
            return

        title, description, hashtags, _, accepted = self._show_upload_dialog("YouTube")
        if not accepted:
            return

        self._run_social_upload_via_mode(
            platform_name="YouTube",
            video_path=video_path,
            caption=self._compose_social_text(description, hashtags),
            title=title,
        )

    def upload_selected_to_facebook(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        title, description, hashtags, _, accepted = self._show_upload_dialog("Facebook")
        if not accepted:
            return
        
        access_token = self.facebook_access_token.text().strip()
        page_id = self.facebook_page_id.text().strip()
        if not access_token or len(access_token) < 50:  # rough check for validity
            QMessageBox.warning(self, "Auth Error", "Invalid or missing Facebook access token.")
            return
        self._upload_video_to_facebook_page(page_id, access_token, video_path, title, description)
        

    def upload_selected_to_instagram(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        selected_video = self.videos[index]
        source_url = str(selected_video.get("source_url") or "").strip()
        if not source_url.startswith(("http://", "https://")):
            QMessageBox.warning(
                self,
                "Instagram Upload Requires URL",
                "Instagram Graph API video publishing requires a publicly reachable HTTP(S) URL. "
                "This selected video only exists locally.",
            )
            return

        _, caption, hashtags, _, accepted = self._show_upload_dialog("Instagram", title_enabled=False)
        if not accepted:
            return

        self._start_upload(
            platform_name="Instagram",
            upload_fn=upload_video_to_instagram_reels,
            upload_kwargs={
                "ig_user_id": self.instagram_business_id.text().strip(),
                "access_token": self.instagram_access_token.text().strip(),
                "video_url": source_url,
                "caption": self._compose_social_text(caption, hashtags),
            },
            success_dialog_title="Instagram Upload Complete",
            success_prefix="Media published successfully. ID:",
        )

    def upload_selected_to_tiktok(self) -> None:
        index = self._selected_video_index()
        if index < 0 or index >= len(self.videos):
            QMessageBox.warning(self, "No Video Selected", "Select a video to upload first.")
            return

        video_path = self.videos[index]["video_file_path"]
        _, caption, hashtags, _, accepted = self._show_upload_dialog("TikTok", title_enabled=False)
        if not accepted:
            return

        video_size = os.path.getsize(video_path)  # MUST be exact bytes!
        MIN_CHUNK = 5 * 1024 * 1024
        MAX_CHUNK = 64 * 1024 * 1024

        raw_token = self.tiktok_access_token.text().strip()
        if isinstance(raw_token, (tuple, list)):
            access_token = raw_token[0] if raw_token else ""
        else:
            access_token = raw_token

        if not access_token:
            QMessageBox.warning(self, "Auth Error", "No valid TikTok access token found.")
            return

        print(f"DEBUG: video_size={video_size} bytes (~{video_size / (1024*1024):.2f} MB)")
        masked_token = f"{access_token[:10]}...{access_token[-6:]}" if len(access_token) > 20 else access_token
        print(f"DEBUG: access token={masked_token}")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

        init_url_direct = "https://open.tiktokapis.com/v2/post/publish/video/init/"
        init_url_inbox = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"

        strategy_candidates: list[tuple[int, int, str]] = []
        if video_size <= MAX_CHUNK:
            strategy_candidates.append((video_size, 1, "single_chunk"))
        else:
            ceil_count = max(1, math.ceil(video_size / MAX_CHUNK))
            even_chunk = max(MIN_CHUNK, math.ceil(video_size / ceil_count))
            strategy_candidates.extend([
                (even_chunk, ceil_count, "even_ceil_count"),
                (64 * 1024 * 1024, math.ceil(video_size / (64 * 1024 * 1024)), "fixed_64mb"),
                (32 * 1024 * 1024, math.ceil(video_size / (32 * 1024 * 1024)), "fixed_32mb"),
                (16 * 1024 * 1024, math.ceil(video_size / (16 * 1024 * 1024)), "fixed_16mb"),
                (8 * 1024 * 1024, math.ceil(video_size / (8 * 1024 * 1024)), "fixed_8mb"),
            ])

        normalized_candidates: list[tuple[int, int, str]] = []
        seen_candidates: set[tuple[int, int]] = set()
        for raw_chunk_size, _raw_total_count, strategy_name in strategy_candidates:
            chunk_size = max(1, int(raw_chunk_size))
            if video_size > MAX_CHUNK:
                chunk_size = min(MAX_CHUNK, max(MIN_CHUNK, chunk_size))
            total_chunk_count = max(1, math.ceil(video_size / chunk_size))
            key = (chunk_size, total_chunk_count)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            normalized_candidates.append((chunk_size, total_chunk_count, strategy_name))

        strategy_candidates = normalized_candidates

        def _extract_uploaded_end_from_headers(response: requests.Response) -> int | None:
            header_value = str(response.headers.get("Range") or response.headers.get("Content-Range") or "").strip()
            if not header_value:
                return None
            match = re.search(r"(\d+)-(\d+)", header_value)
            if not match:
                return None
            try:
                end_value = int(match.group(2))
            except (TypeError, ValueError):
                return None
            return end_value if end_value >= 0 else None

        last_error: Exception | None = None
        publish_id: str | None = None
        upload_succeeded = False

        for chunk_size, total_chunk_count, strategy_name in strategy_candidates:
            upload_chunk_count = max(1, math.ceil(video_size / chunk_size))
            print(
                f"DEBUG: strategy={strategy_name} chunk_size={chunk_size} "
                f"total_chunk_count={total_chunk_count} upload_chunk_count={upload_chunk_count}"
            )

            source_info = {
                "source": "FILE_UPLOAD",

                "chunk_size": chunk_size,
                
            }
            payload = {
                "source_info": source_info,
                "post_info": {
                    "title": self._compose_social_text(caption, hashtags),
                    "privacy_level": str(self.tiktok_privacy_level.currentData() or "PUBLIC_TO_EVERYONE"),
                    "disable_comment": False,
                    "disable_duet": False,
                    "disable_stitch": False,
                },
            }

            # Direct post init is generally more permissive for chunk metadata.
            init_modes: list[tuple[str, str]] = [("direct_post", init_url_direct)]
            if total_chunk_count == 1:
                init_modes.append(("inbox_draft", init_url_inbox))

            for init_mode, init_url in init_modes:
                print(f"DEBUG: init_mode={init_mode} init_url={init_url}")
                resp = requests.post(init_url, headers=headers, json=payload)
                try:
                    init_data = resp.json()
                except ValueError:
                    init_data = {"raw": resp.text[:500]}

                if resp.status_code != 200:
                    print("Status check failed:", init_data)
                    last_error = RuntimeError(
                        f"TikTok init failed ({strategy_name}, {init_mode}): {resp.status_code} - {init_data}"
                    )
                    continue

                print("Status response:", init_data)
                upload_url = str(init_data.get("data", {}).get("upload_url") or "")
                publish_id = init_data.get("data", {}).get("publish_id")
                if not upload_url or not publish_id:
                    last_error = RuntimeError(
                        f"TikTok init missing upload_url/publish_id ({strategy_name}, {init_mode}): {init_data}"
                    )
                    continue

                try:
                    upload_resp = None
                    bytes_uploaded = 0
                    with open(video_path, "rb") as f:
                        chunk_index = 0
                        while bytes_uploaded < video_size:
                            f.seek(bytes_uploaded)
                            chunk = f.read(min(chunk_size, video_size - bytes_uploaded))
                            if not chunk:
                                break

                            start_byte = bytes_uploaded
                            end_byte = start_byte + len(chunk) - 1
                            chunk_index += 1
                            upload_headers = {
                                "Content-Type": "video/mp4",
                                "Content-Length": str(len(chunk)),
                                "Content-Range": f"bytes {start_byte}-{end_byte}/{video_size}",
                            }

                            print(
                                f"DEBUG: Uploading chunk {chunk_index}/{upload_chunk_count} "
                                f"({len(chunk)} bytes) range: bytes {start_byte}-{end_byte}/{video_size}"
                            )
                            upload_resp = requests.put(upload_url, headers=upload_headers, data=chunk)
                            print(f"DEBUG: Chunk upload status code: {upload_resp.status_code}")

                            if upload_resp.text.strip():
                                try:
                                    print("DEBUG: Chunk upload response:", upload_resp.json())
                                except ValueError:
                                    print("DEBUG: Chunk upload response is not JSON:", upload_resp.text[:500])

                            server_uploaded_end = _extract_uploaded_end_from_headers(upload_resp)
                            if upload_resp.status_code == 416:
                                if server_uploaded_end is not None and server_uploaded_end + 1 > bytes_uploaded:
                                    bytes_uploaded = min(video_size, server_uploaded_end + 1)
                                    print(
                                        f"DEBUG: Recovered from HTTP 416 using server range; continuing at offset {bytes_uploaded}."
                                    )
                                    continue
                                raise RuntimeError(
                                    f"Upload failed ({strategy_name}, {init_mode}): {upload_resp.status_code} - {upload_resp.text}"
                                )

                            if upload_resp.status_code not in (200, 201, 202, 204, 206):
                                raise RuntimeError(
                                    f"Upload failed ({strategy_name}, {init_mode}): {upload_resp.status_code} - {upload_resp.text}"
                                )

                            next_offset = start_byte + len(chunk)
                            if server_uploaded_end is not None:
                                next_offset = max(next_offset, server_uploaded_end + 1)
                            bytes_uploaded = min(video_size, next_offset)

                    if upload_resp is None or bytes_uploaded < video_size:
                        raise RuntimeError(
                            f"Upload failed ({strategy_name}, {init_mode}): sent {bytes_uploaded} bytes, expected {video_size} bytes."
                        )

                    upload_succeeded = True
                    break
                except RuntimeError as upload_error:
                    last_error = upload_error
                    print(
                        f"DEBUG: strategy {strategy_name} mode={init_mode} failed, retrying with next init/strategy. error={upload_error}"
                    )
                    continue

            if upload_succeeded:
                break

        if not upload_succeeded:
            if last_error:
                raise RuntimeError(str(last_error))
            raise RuntimeError("TikTok upload failed: no chunking strategy succeeded.")

        if not publish_id:
            raise RuntimeError("TikTok upload succeeded but publish_id was not returned.")

        self.check_tiktok_status(access_token, publish_id)
        self._append_log("TikTok Upload complete")
        return

    def check_tiktok_status(self, access_token, publish_id):
        url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {"publish_id": publish_id}
        
        resp = requests.post(url, headers=headers, json=payload)
        
        if resp.status_code == 200:
            data = resp.json()
            print("Status response:", data)
            # Look for data['data']['status'] — possible values:
            # "PROCESSING_UPLOAD", "UPLOAD_SUCCESS", "POSTED", "FAILED", etc.
            return data
        else:
            print("Status check failed:", resp.text)

    def _cancel_social_upload_run(self, platform_name: str, reason: str = "") -> None:
        pending = self.social_upload_pending.pop(platform_name, None)
        timer = self.social_upload_timers.get(platform_name)
        if timer is not None:
            timer.stop()

        if isinstance(pending, dict):
            relay_future = pending.get("cdp_relay_future")
            if isinstance(relay_future, concurrent.futures.Future) and not relay_future.done():
                relay_future.cancel()

        status_label = self.social_upload_status_labels.get(platform_name)
        progress_bar = self.social_upload_progress_bars.get(platform_name)
        if status_label is not None:
            status_suffix = f" ({reason})" if reason else ""
            status_label.setText(f"Status: restarted automation{status_suffix}.")
        if progress_bar is not None:
            progress_bar.setVisible(False)
            progress_bar.setValue(0)

    def _start_social_browser_upload(self, platform_name: str, video_path: str, caption: str, title: str) -> None:
        if not self._is_browser_tab_enabled(platform_name):
            QMessageBox.warning(self, "Social Upload", f"{platform_name} upload tab is disabled. Re-enable it from View → Browser Tabs.")
            return

        if platform_name in self.social_upload_pending:
            self._append_log(f"{platform_name}: previous automation run detected; canceling and starting fresh.")
            self._cancel_social_upload_run(platform_name, reason="new run requested")

        browser = self.social_upload_browsers.get(platform_name)
        timer = self.social_upload_timers.get(platform_name)
        status_label = self.social_upload_status_labels.get(platform_name)
        progress_bar = self.social_upload_progress_bars.get(platform_name)
        tab_index = self.social_upload_tab_indices.get(platform_name)
        if not browser or not timer or not status_label or not progress_bar or tab_index is None:
            QMessageBox.warning(self, "Social Upload", f"{platform_name} upload tab is not initialized.")
            return

        video_file = Path(str(video_path))
        encoded_video = ""
        if video_file.exists() and video_file.is_file():
            # Avoid embedding very large TikTok blobs into in-page JS payloads.
            # Large base64 payloads can freeze the browser process before upload starts.
            tiktok_inline_limit_bytes = 200 * 1024 * 1024
            should_inline_video = (
                platform_name != "TikTok"
                or video_file.stat().st_size <= tiktok_inline_limit_bytes
            )
            if should_inline_video:
                try:
                    encoded_video = base64.b64encode(video_file.read_bytes()).decode("ascii")
                except Exception:
                    encoded_video = ""
            else:
                # Keep a tiny payload available for lightweight synthetic input probes.
                encoded_video = "AA=="

        self.social_upload_pending[platform_name] = {
            "platform": platform_name,
            "video_path": str(video_path),
            "caption": str(caption or ""),
            "title": str(title or ""),
            "attempts": 0,
            "allow_file_dialog": True,
            "video_base64": encoded_video,
            "video_name": video_file.name or "upload.mp4",
            "video_mime": "video/mp4",
            "youtube_options": dict(getattr(self, "youtube_browser_upload_options", {})),
        }
        self._append_log(
            f"{platform_name}: queued browser upload video path={video_path} (exists={Path(str(video_path)).exists()})"
        )

        progress_bar.setVisible(True)
        progress_bar.setValue(10)
        status_label.setText("Status: staging video in current tab...")
        self._append_log(
            f"Browser-based {platform_name} upload: using current social tab and attempting automated file staging."
        )
        self.browser_tabs.setCurrentIndex(tab_index)
        self._run_social_browser_upload_step(platform_name)

    def _try_social_upload_step_via_cdp_relay(
        self,
        platform_name: str,
        pending: dict,
        browser: QWebEngineView,
        attempts: int,
        status_label: QLabel,
        progress_bar: QProgressBar,
        timer: QTimer,
    ) -> bool:
        if not self.cdp_social_upload_relay_enabled.isChecked():
            return False
        if self._cdp_relay_temporarily_disabled:
            status_label.setText("Status: CDP relay paused for this session. Toggle CDP Relay Mode off/on to retry.")
            timer.start(1500)
            return True

        relay_url = self.cdp_social_upload_relay_url.text().strip()
        if not relay_url:
            status_label.setText("Status: CDP relay enabled but URL is empty. Set CDP Relay URL in App Preferences.")
            timer.start(1500)
            return True

        payload = {
            "platform": platform_name.lower(),
            "attempt": attempts,
            "current_url": browser.url().toString().strip(),
            "video_path": str(pending.get("video_path") or ""),
            "video_name": str(pending.get("video_name") or "upload.mp4"),
            "video_mime": str(pending.get("video_mime") or "video/mp4"),
            "caption": str(pending.get("caption") or ""),
            "title": str(pending.get("title") or ""),
            "youtube_options": pending.get("youtube_options") or {},
            "qtwebengine_remote_debugging": os.getenv("QTWEBENGINE_REMOTE_DEBUGGING", "").strip(),
        }

        relay_future = pending.get("cdp_relay_future")
        if isinstance(relay_future, concurrent.futures.Future):
            if not relay_future.done():
                status_label.setText("Status: waiting for CDP relay response...")
                timer.start(250)
                return True

            pending.pop("cdp_relay_future", None)
            try:
                relay_result = relay_future.result()
            except Exception as exc:
                relay_result = {
                    "kind": "error",
                    "error": str(exc),
                }
        else:
            pending["cdp_relay_future"] = self._cdp_relay_executor.submit(
                self._request_cdp_relay_step,
                relay_url,
                payload,
            )
            status_label.setText("Status: CDP relay step running...")
            timer.start(250)
            return True

        result_kind = str(relay_result.get("kind") or "error")
        if result_kind == "timeout":
            exc = relay_result.get("error", "timeout")
            status_label.setText("Status: CDP relay request timed out; retrying...")
            if not pending.get("cdp_relay_timeout_logged"):
                self._append_log(
                    f"WARNING: {platform_name} CDP relay request timed out ({exc}); keeping relay mode active and retrying."
                )
                pending["cdp_relay_timeout_logged"] = True
            timer.start(1500)
            return True
        if result_kind != "ok":
            exc = relay_result.get("error", "unknown relay error")
            self._cdp_relay_temporarily_disabled = True
            status_label.setText("Status: CDP relay unavailable; retry paused for this session.")
            if not pending.get("cdp_relay_error_logged"):
                self._append_log(
                    f"WARNING: {platform_name} CDP relay unavailable ({exc}); CDP relay mode is active, so DOM fallback is disabled "
                    "for this upload and relay attempts are paused for this session."
                )
                self._append_log(
                    "TIP: Start your CDP relay service and toggle CDP Relay Mode off/on (or restart app) to retry relay mode."
                )
                pending["cdp_relay_error_logged"] = True
            timer.start(1500)
            return True

        relay_data = relay_result.get("relay_data")

        if not isinstance(relay_data, dict):
            relay_data = {}

        handled = bool(relay_data.get("handled", True))
        if not handled:
            relay_status = str(relay_data.get("status") or "").strip()
            status_label.setText(relay_status or f"Status: CDP relay step {attempts} not handled yet.")

            if "cdp unavailable" in relay_status.lower() or "browser.setdownloadbehavior" in relay_status.lower():
                self._cdp_relay_temporarily_disabled = True
                if not pending.get("cdp_relay_fallback_logged"):
                    self._append_log(
                        f"WARNING: {platform_name} CDP relay reported unsupported target ({relay_status or 'unavailable'}); "
                        "disabling relay for this session and continuing with built-in DOM upload automation."
                    )
                    pending["cdp_relay_fallback_logged"] = True
                return False

            retry_ms = int(relay_data.get("retry_ms", 1500) or 1500)
            timer.start(max(400, retry_ms))
            return True

        progress = int(relay_data.get("progress", min(95, 20 + attempts * 6)) or 0)
        progress_bar.setVisible(True)
        progress_bar.setValue(max(0, min(100, progress)))
        status_label.setText(str(relay_data.get("status") or f"Status: CDP relay step {attempts}."))

        relay_log = str(relay_data.get("log") or "").strip()
        if relay_log:
            self._append_log(f"{platform_name} CDP relay: {relay_log}")

        if bool(relay_data.get("done")):
            self.social_upload_pending.pop(platform_name, None)
            return True

        retry_ms = int(relay_data.get("retry_ms", 1500) or 1500)
        timer.start(max(400, retry_ms))
        return True

    def _request_cdp_relay_step(self, relay_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(relay_url, json=payload, timeout=CDP_RELAY_TIMEOUT_SECONDS)
            relay_data = response.json() if response.content else {}
            if not response.ok:
                return {
                    "kind": "http_error",
                    "error": f"HTTP {response.status_code}: {str(relay_data)[:300]}",
                }
            return {
                "kind": "ok",
                "relay_data": relay_data,
            }
        except requests.exceptions.ReadTimeout as exc:
            return {
                "kind": "timeout",
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "kind": "error",
                "error": str(exc),
            }

    def _run_social_browser_upload_step(self, platform_name: str) -> None:
        pending = self.social_upload_pending.get(platform_name)
        browser = self.social_upload_browsers.get(platform_name)
        timer = self.social_upload_timers.get(platform_name)
        status_label = self.social_upload_status_labels.get(platform_name)
        progress_bar = self.social_upload_progress_bars.get(platform_name)
        if not pending or not browser or not timer or not status_label or not progress_bar:
            return

        pending["attempts"] = int(pending.get("attempts", 0)) + 1
        attempts = int(pending["attempts"])

        if platform_name == "Instagram":
            max_attempts = 6
        elif platform_name == "TikTok":
            max_attempts = 12
        elif platform_name == "Facebook":
            max_attempts = 8
        elif platform_name == "YouTube":
            max_attempts = 24
        else:
            max_attempts = 2
        if attempts > max_attempts:
            status_label.setText("Status: automation timed out; finish manually in this tab.")
            progress_bar.setVisible(False)
            self._append_log(
                f"WARNING: {platform_name} browser automation timed out after {attempts - 1} attempts."
            )
            self.social_upload_pending.pop(platform_name, None)
            return
            

        if self._try_social_upload_step_via_cdp_relay(platform_name, pending, browser, attempts, status_label, progress_bar, timer):
            return

        payload_json = json.dumps(
            {
                "caption": str(pending.get("caption") or ""),
                "title": str(pending.get("title") or ""),
                "platform": platform_name.lower(),
                "video_path": str(pending.get("video_path") or ""),
                "video_base64": str(pending.get("video_base64") or ""),
                "video_name": str(pending.get("video_name") or "upload.mp4"),
                "video_mime": str(pending.get("video_mime") or "video/mp4"),
                "allow_file_dialog": bool(pending.get("allow_file_dialog", False)),
                "youtube_options": pending.get("youtube_options") or {},
            },
            ensure_ascii=True,
        )
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("ascii")
        script = """
            (() => {
                try {
                    const payload = JSON.parse(atob("__PAYLOAD_B64__"));
                    const norm = (s) => String(s || "").toLowerCase();
                    const platform = norm(payload.platform);
                    const pick = (arr) => arr.find(Boolean) || null;
                    const collectDeep = (selector) => {
                        const results = [];
                        const seen = new Set();
                        const roots = [document];
                        while (roots.length) {
                            const root = roots.pop();
                            let nodes = [];
                            try { nodes = Array.from(root.querySelectorAll(selector)); } catch (_) {}
                            for (const node of nodes) {
                                if (!seen.has(node)) {
                                    seen.add(node);
                                    results.push(node);
                                }
                            }
                            let all = [];
                            try { all = Array.from(root.querySelectorAll('*')); } catch (_) {}
                            for (const el of all) {
                                if (el && el.shadowRoot) roots.push(el.shadowRoot);
                            }
                        }
                        return results;
                    };
                    const isVisible = (node) => Boolean(node && (node.offsetWidth || node.offsetHeight || node.getClientRects().length));
                    const firstVisibleNode = (nodes) => nodes.find((node) => isVisible(node)) || null;
                    const bySelectors = (selectors, options = {}) => {
                        const contexts = (options.contexts && options.contexts.length)
                            ? options.contexts
                            : [document];
                        for (const selector of selectors) {
                            const matches = [];
                            for (const contextNode of contexts) {
                                let nodes = [];
                                try {
                                    nodes = Array.from(contextNode.querySelectorAll(selector));
                                } catch (_) {}
                                matches.push(...nodes);
                            }
                            const visible = firstVisibleNode(matches);
                            if (visible) return visible;
                            if (matches.length) return matches[0];
                        }
                        return null;
                    };
                    const normalizedNodeText = (node) => [
                        norm(node.innerText || node.textContent),
                        norm(node.getAttribute("aria-label")),
                        norm(node.getAttribute("title")),
                        norm(node.getAttribute("data-testid")),
                    ].join(" ");
                    const injectClickActions = (target) => {
                        if (!target) return false;
                        try { target.scrollIntoView({ block: "center", inline: "center", behavior: "instant" }); } catch (_) {}
                        const events = [
                            ["pointerdown", PointerEvent],
                            ["mousedown", MouseEvent],
                            ["pointerup", PointerEvent],
                            ["mouseup", MouseEvent],
                            ["click", MouseEvent],
                        ];
                        let dispatched = false;
                        for (const [eventName, EventCtor] of events) {
                            try {
                                const ev = new EventCtor(eventName, { bubbles: true, cancelable: true, composed: true, button: 0, buttons: 1 });
                                target.dispatchEvent(ev);
                                dispatched = true;
                            } catch (_) {}
                        }
                        try {
                            if (typeof target.click === "function") {
                                target.click();
                                dispatched = true;
                            }
                        } catch (_) {}
                        return dispatched;
                    };
                    const clickNodeOrAncestor = (node) => {
                        if (!node) return false;
                        const candidates = [
                            node,
                            node.closest && node.closest('button, [role="button"], a, [role="link"], div[tabindex], div'),
                            node.parentElement,
                        ].filter(Boolean);
                        for (const candidate of candidates) {
                            if (injectClickActions(candidate)) return true;
                        }
                        return false;
                    };
                    const clickNodeSingle = (node) => {
                        if (!node) return false;
                        try { node.scrollIntoView({ block: "center", inline: "center", behavior: "instant" }); } catch (_) {}
                        try {
                            if (typeof node.click === "function") {
                                node.click();
                                return true;
                            }
                        } catch (_) {}
                        try {
                            const ev = new MouseEvent("click", { bubbles: true, cancelable: true, composed: true, button: 0, buttons: 1 });
                            node.dispatchEvent(ev);
                            return true;
                        } catch (_) {}
                        return false;
                    };
                    const findClickableByHints = (hints, options = {}) => {
                        const normalizedHints = hints.map((hint) => norm(hint)).filter(Boolean);
                        const excludeHints = (options.excludeHints || []).map((hint) => norm(hint)).filter(Boolean);
                        const contexts = (options.contexts && options.contexts.length)
                            ? options.contexts
                            : [document];
                        const clickableNodes = [];
                        for (const contextNode of contexts) {
                            let nodes = [];
                            try {
                                nodes = Array.from(contextNode.querySelectorAll('button, [role="button"], a, [role="link"], div[tabindex], span, div'));
                            } catch (_) {}
                            clickableNodes.push(...nodes);
                        }
                        const isValidMatch = (text) => {
                            if (!normalizedHints.some((hint) => text.includes(hint))) return false;
                            if (excludeHints.length && excludeHints.some((hint) => text.includes(hint))) return false;
                            return true;
                        };
                        for (const node of clickableNodes) {
                            if (!isVisible(node)) continue;
                            const text = normalizedNodeText(node);
                            if (isValidMatch(text)) return node;
                        }
                        for (const node of clickableNodes) {
                            const text = normalizedNodeText(node);
                            if (isValidMatch(text)) return node;
                        }
                        return null;
                    };

                    let openUploadClicked = false;
                    const requestedVideoPath = String(payload.video_path || payload.videoPath || "");
                    const videoBase64 = String(payload.video_base64 || "");
                    const videoName = String(payload.video_name || "upload.mp4");
                    const videoMime = String(payload.video_mime || "video/mp4");
                    const allowFileDialog = Boolean(payload.allow_file_dialog);
                    const titleText = String(payload.title || "").trim();
                    const captionText = String(payload.caption || "").trim();
                    const captionRequired = (platform === "facebook" || platform === "instagram") && Boolean(captionText);
                    const uploadState = window.__codexSocialUploadState = window.__codexSocialUploadState || {};
                    const instagramState = uploadState.instagram = uploadState.instagram || {};
                    const facebookState = uploadState.facebook = uploadState.facebook || {};
                    const youtubeState = uploadState.youtube = uploadState.youtube || {};
                    if (platform === "instagram") {
                        const instagramDialog = bySelectors(['div[role="dialog"][aria-label*="create new post" i]']);
                        if (instagramDialog) {
                            instagramState.dialogSeen = true;
                            instagramState.postClicked = true;
                        } else {
                            if (!instagramState.createClicked) {
                                const createSpanButton = pick(collectDeep('span.html-span[aria-describedby="_r_m_"], span[aria-describedby="_r_m_"], span.html-span[aria-describedby], span[aria-describedby^="_r_"]'));
                                const createButton = createSpanButton
                                    || bySelectors([
                                        'div[role="button"][tabindex="0"]',
                                        'button',
                                        'a[role="link"]',
                                    ])
                                    || findClickableByHints(["create", "new post"]);
                                if (createButton) {
                                    const clicked = clickNodeOrAncestor(createButton);
                                    openUploadClicked = clicked || openUploadClicked;
                                    if (clicked) instagramState.createClicked = true;
                                }
                            } else if (!instagramState.dialogSeen) {
                                const menuContexts = [
                                    ...collectDeep('div[role="menu"]'),
                                    ...collectDeep('div[role="dialog"]:not([aria-label*="create new post" i])'),
                                ];
                                const postButton = pick(
                                    menuContexts.flatMap((ctx) => {
                                        try {
                                            return Array.from(ctx.querySelectorAll('a[href="#"][role="link"][tabindex="0"] > div.html-div, a[href="#"][role="link"][tabindex="0"], a[role="link"][href="#"][tabindex="0"]'));
                                        } catch (_) {
                                            return [];
                                        }
                                    })
                                ) || findClickableByHints(["post"], { contexts: menuContexts, excludeHints: ["reel"] });
                                if (postButton) {
                                    const postText = normalizedNodeText(postButton);
                                    if (postText.includes("post") && !postText.includes("reel")) {
                                        const clicked = clickNodeOrAncestor(postButton);
                                        openUploadClicked = clicked || openUploadClicked;
                                        if (clicked) instagramState.postClicked = true;
                                    }
                                }
                            }
                        }
                    }

                    if (platform === "facebook" && !facebookState.createPostOpened) {
                        const facebookCreatePostButton = findClickableByHints(["create post", "what's on your mind"]);
                        if (facebookCreatePostButton) {
                            const clicked = clickNodeOrAncestor(facebookCreatePostButton);
                            openUploadClicked = clicked || openUploadClicked;
                            if (clicked) {
                                facebookState.createPostOpened = true;
                            }
                        }
                    }

                    const setTextValue = (node, value) => {
                        if (!node) return false;
                        const text = String(value || "");
                        try {
                            if ("value" in node) {
                                node.focus();
                                node.value = text;
                                node.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
                                node.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
                                return true;
                            }
                        } catch (_) {}
                        try {
                            if (node.isContentEditable || node.getAttribute("contenteditable") === "true") {
                                node.focus();
                                node.textContent = text;
                                node.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertText" }));
                                node.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
                                return true;
                            }
                        } catch (_) {}
                        return false;
                    };
                    const findTextInputTarget = () => {
                        const selectors = [
                            'textarea[aria-label*="caption" i]',
                            'textarea[placeholder*="caption" i]',
                            'textarea[aria-label*="write" i]',
                            'textarea[placeholder*="write" i]',
                            'div[role="textbox"][contenteditable="true"]',
                            'div[contenteditable="true"][aria-label*="write" i]',
                            'div[contenteditable="true"][aria-label*="post" i]',
                            'textarea',
                        ];
                        const dialogRoots = collectDeep('div[role="dialog"]');
                        for (const dialog of dialogRoots) {
                            if (!dialog) continue;
                            for (const selector of selectors) {
                                let nodes = [];
                                try { nodes = Array.from(dialog.querySelectorAll(selector)); } catch (_) {}
                                const visibleMatch = nodes.find((node) => isVisible(node));
                                if (visibleMatch) return visibleMatch;
                                if (nodes.length) return nodes[0];
                            }
                        }
                        return bySelectors(selectors);
                    };

                    let textFilled = false;
                    let captionReady = !captionRequired;
                    if (platform === "facebook" && captionRequired) {
                        const textTarget = findTextInputTarget();
                        textFilled = setTextValue(textTarget, captionText);
                        captionReady = textFilled;
                    }

                    if (platform === "facebook" && captionReady && !facebookState.uploadChooserOpened) {
                        const facebookUploadButton = findClickableByHints([
                            "photo/video",
                            "photo or video",
                            "add photo",
                            "add video",
                        ]);
                        if (facebookUploadButton) {
                            const clicked = clickNodeOrAncestor(facebookUploadButton);
                            openUploadClicked = clicked || openUploadClicked;
                            if (clicked) {
                                facebookState.uploadChooserOpened = true;
                            }
                        }
                    }

                    const fileInputs = collectDeep('input[type="file"]');
                    const pickVideoInput = () => {
                        if (platform === "instagram") {
                            const instagramDialog = bySelectors(['div[role="dialog"][aria-label*="create new post" i]']);
                            if (!instagramDialog) return null;

                            const formInputs = Array.from(instagramDialog.querySelectorAll('form[enctype="multipart/form-data" i][method="post" i][role="presentation"] input[type="file"]'));
                            const exactInstagramInput = formInputs.find((node) => {
                                const accept = norm(node.getAttribute("accept"));
                                const className = norm(node.className);
                                return accept.includes("video/mp4") && accept.includes("video/quicktime") && className.includes("x1s85apg");
                            });
                            if (exactInstagramInput) return exactInstagramInput;
                            return null;
                        }
                        if (platform === "facebook") {
                            const byFacebookAccept = fileInputs.find((node) => {
                                const accept = norm(node.getAttribute("accept"));
                                return accept.includes("video/*") && accept.includes("image/*");
                            });
                            if (byFacebookAccept) return byFacebookAccept;
                            const byFacebookClass = fileInputs.find((node) => norm(node.className).includes("x1s85apg"));
                            if (byFacebookClass) return byFacebookClass;
                        }
                        const byExactInstagramAccept = fileInputs.find((node) => {
                            const accept = norm(node.getAttribute("accept"));
                            return accept.includes("video/mp4") || accept.includes("video/quicktime") || accept.includes("video/*");
                        });
                        if (byExactInstagramAccept) return byExactInstagramAccept;
                        const byAcceptVideo = fileInputs.find((node) => norm(node.getAttribute("accept")).includes("video"));
                        if (byAcceptVideo) return byAcceptVideo;
                        const byClassVideo = fileInputs.find((node) => norm(node.className).includes("video") || norm(node.className).includes("x1s85apg"));
                        if (byClassVideo) return byClassVideo;
                        return pick(fileInputs);
                    };
                    const fileInput = pickVideoInput();
                    const setInputFiles = (input, files) => {
                        try {
                            const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "files");
                            if (descriptor && typeof descriptor.set === "function") {
                                descriptor.set.call(input, files);
                                return true;
                            }
                        } catch (_) {}
                        try {
                            Object.defineProperty(input, "files", { value: files, configurable: true });
                            return true;
                        } catch (_) {
                            return false;
                        }
                    };
                    let fileDialogTriggered = false;
                    let fakeProbeInjected = false;
                    if (fileInput && (platform !== "facebook" || captionReady)) {
                        if (requestedVideoPath) {
                            try { fileInput.setAttribute("data-codex-video-path", requestedVideoPath); } catch (_) {}
                        }
                        fileInput.style.display = "block";
                        fileInput.style.visibility = "visible";
                        fileInput.removeAttribute("hidden");
                        try { fileInput.removeAttribute("disabled"); } catch (_) {}

                        const alreadyHasFile = Boolean(fileInput.files && fileInput.files.length > 0);
                        const alreadyStaged = platform === "facebook" ? Boolean(facebookState.fileStaged) : false;
                        const shouldInjectDirectly = platform !== "tiktok" || videoBase64 !== "AA==";
                        if (!alreadyHasFile && !alreadyStaged && videoBase64 && shouldInjectDirectly) {
                            try {
                                const binary = atob(videoBase64);
                                const bytes = new Uint8Array(binary.length);
                                for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
                                const file = new File([bytes], videoName, { type: videoMime });
                                const dt = new DataTransfer();
                                dt.items.add(file);
                                if (setInputFiles(fileInput, dt.files)) {
                                    fileInput.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
                                    fileInput.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
                                    fileDialogTriggered = true;
                                    if (platform === "facebook") {
                                        facebookState.fileStaged = true;
                                    }
                                }
                            } catch (_) {}
                        }

                        if (platform === "tiktok" && videoBase64 === "AA==") {
                            try {
                                const binary = atob(videoBase64);
                                const bytes = new Uint8Array(binary.length);
                                for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
                                const probeFile = new File([bytes], "codex-probe.bin", { type: "application/octet-stream" });
                                const existingProbe = document.getElementById("codex-fake-upload-probe");
                                if (existingProbe && existingProbe.parentElement) existingProbe.parentElement.removeChild(existingProbe);
                                const probeInput = document.createElement("input");
                                probeInput.type = "file";
                                probeInput.id = "codex-fake-upload-probe";
                                probeInput.style.display = "none";
                                document.body.appendChild(probeInput);
                                const probeDt = new DataTransfer();
                                probeDt.items.add(probeFile);
                                fakeProbeInjected = setInputFiles(probeInput, probeDt.files);

                                const promptId = "codex-tiktok-user-interaction-required";
                                const tiktokOverlayState = uploadState.tiktok = uploadState.tiktok || {};
                                const shouldShowUserPrompt = !Boolean(tiktokOverlayState.userInteractionConfirmed);
                                const existingPrompt = document.getElementById(promptId);
                                if (!shouldShowUserPrompt) {
                                    if (existingPrompt && existingPrompt.parentElement) existingPrompt.parentElement.removeChild(existingPrompt);
                                } else if (!existingPrompt) {
                                    const userPrompt = document.createElement("div");
                                    userPrompt.id = promptId;
                                    userPrompt.setAttribute("role", "button");
                                    userPrompt.tabIndex = 0;
                                    userPrompt.style.display = "inline-block";
                                    userPrompt.style.marginLeft = "12px";
                                    userPrompt.style.verticalAlign = "middle";
                                    userPrompt.style.maxWidth = "min(520px, 70vw)";
                                    userPrompt.style.padding = "10px 14px";
                                    userPrompt.style.fontSize = "16px";
                                    userPrompt.style.fontWeight = "700";
                                    userPrompt.style.lineHeight = "1.35";
                                    userPrompt.style.textAlign = "left";
                                    userPrompt.style.cursor = "pointer";
                                    userPrompt.style.textDecoration = "none";
                                    userPrompt.style.border = "2px solid #ff4d4f";
                                    userPrompt.style.borderRadius = "8px";
                                    userPrompt.style.boxShadow = "0 4px 10px rgba(0, 0, 0, 0.18)";
                                    userPrompt.style.background = "rgba(255, 255, 255, 0.98)";
                                    userPrompt.style.color = "#b00020";
                                    userPrompt.textContent = "User Interaction Required, Click here to continue!";
                                    const activateFileInput = () => {
                                        try {
                                            const tiktokUploadState = window.__codexSocialUploadState = window.__codexSocialUploadState || {};
                                            const tiktokPhaseState = tiktokUploadState.tiktok = tiktokUploadState.tiktok || {};
                                            tiktokPhaseState.userInteractionConfirmed = true;
                                            tiktokPhaseState.awaitingDraftAfterUserGesture = true;
                                            tiktokPhaseState.lastActionAtMs = Date.now();
                                        } catch (_) {}
                                        try { userPrompt.style.display = "none"; } catch (_) {}
                                        try { userPrompt.remove(); } catch (_) {}
                                        try { fileInput.scrollIntoView({ block: "center", inline: "center", behavior: "instant" }); } catch (_) {}
                                        try { fileInput.click(); } catch (_) {}
                                    };
                                    userPrompt.addEventListener("click", activateFileInput);
                                    userPrompt.addEventListener("keydown", (ev) => {
                                        if (ev.key === "Enter" || ev.key === " ") {
                                            ev.preventDefault();
                                            activateFileInput();
                                        }
                                    });
                                    if (fileInput.parentElement) {
                                        try { fileInput.insertAdjacentElement("afterend", userPrompt); } catch (_) { fileInput.parentElement.appendChild(userPrompt); }
                                    } else {
                                        document.body.appendChild(userPrompt);
                                    }
                                }
                            } catch (_) {}
                        }
                    }

                    const fileReadySignal = Boolean(
                        (fileInput && fileInput.files && fileInput.files.length > 0)
                        || document.querySelector('video')
                        || document.querySelector('[aria-label*="uploaded" i], [aria-label*="uploading" i], progress')
                    );

                    if (platform === "facebook") {
                        if (fileReadySignal) {
                            facebookState.fileStaged = true;
                            if (!facebookState.fileReadyAtMs) {
                                facebookState.fileReadyAtMs = Date.now();
                            }
                        } else {
                            facebookState.fileReadyAtMs = 0;
                        }
                    }
                    const facebookSubmitDelayElapsed = platform !== "facebook"
                        || Boolean(facebookState.fileReadyAtMs && (Date.now() - Number(facebookState.fileReadyAtMs)) >= 2000);

                    let nextClicked = false;
                    let submitClicked = false;
                    if (platform === "instagram" && fileReadySignal) {
                        const instagramDialog = bySelectors(['div[role="dialog"][aria-label*="create new post" i]']) || document;
                        const nextButton = pick(Array.from(instagramDialog.querySelectorAll('div[role="button"][tabindex="0"], button')).filter((node) => normalizedNodeText(node).includes("next"))) || findClickableByHints(["next"]);
                        const nextClicks = Number(instagramState.nextClicks || 0);
                        if (nextButton && nextClicks < 2) {
                            nextClicked = clickNodeOrAncestor(nextButton) || nextClicked;
                            if (nextClicked) {
                                instagramState.nextClicks = nextClicks + 1;
                            }
                        }

                        if (Number(instagramState.nextClicks || 0) >= 2 && captionRequired) {
                            const textTarget = findTextInputTarget();
                            textFilled = setTextValue(textTarget, captionText) || textFilled;
                        }
                        captionReady = !captionRequired || textFilled;

                        if (Number(instagramState.nextClicks || 0) >= 2 && captionReady) {
                            const shareButton = pick(Array.from(instagramDialog.querySelectorAll('div[role="button"][tabindex="0"], button, a')).filter((node) => normalizedNodeText(node).includes("share"))) || findClickableByHints(["share"]);
                            if (shareButton) {
                                submitClicked = clickNodeOrAncestor(shareButton) || submitClicked;
                            }
                        }
                    }

                    if (platform === "facebook" && fileReadySignal && captionReady && facebookSubmitDelayElapsed) {
                        const explicitPostButton = bySelectors(['div[aria-label="Post"][role="button"][tabindex="0"]']);
                        if (explicitPostButton) {
                            submitClicked = clickNodeOrAncestor(explicitPostButton) || submitClicked;
                        }

                        if (!submitClicked) {
                            const submitButton = findClickableByHints(["post", "share"]);
                            if (submitButton) {
                                submitClicked = clickNodeOrAncestor(submitButton) || submitClicked;
                            }
                        }

                        if (!submitClicked) {
                            const submitInput = bySelectors(['input[type="submit"]']);
                            if (submitInput) {
                                const form = submitInput.form || submitInput.closest('form');
                                try {
                                    submitInput.removeAttribute('disabled');
                                    submitInput.style.display = 'block';
                                    submitInput.style.visibility = 'visible';
                                    submitInput.style.opacity = '1';
                                } catch (_) {}
                                try {
                                    if (form && typeof form.requestSubmit === 'function') {
                                        form.requestSubmit(submitInput);
                                        submitClicked = true;
                                    }
                                } catch (_) {}
                                if (!submitClicked) {
                                    try {
                                        submitInput.click();
                                        submitClicked = true;
                                    } catch (_) {}
                                }
                                if (!submitClicked) {
                                    try {
                                        submitInput.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true }));
                                        submitClicked = true;
                                    } catch (_) {}
                                }
                                if (!submitClicked && form) {
                                    try {
                                        form.submit();
                                        submitClicked = true;
                                    } catch (_) {}
                                }
                            }
                        }
                    }

                    let tiktokPostEnabled = false;
                    const tiktokState = uploadState.tiktok = uploadState.tiktok || {};
                    if (platform === "tiktok") {
                        if (tiktokState.lastVideoPath !== requestedVideoPath) {
                            tiktokState.lastVideoPath = requestedVideoPath;
                            tiktokState.lastSubmitAttemptAtMs = 0;
                            tiktokState.lastActionAtMs = 0;
                            tiktokState.captionSetAtMs = 0;
                            tiktokState.submitClicked = false;
                            tiktokState.userInteractionConfirmed = false;
                            tiktokState.awaitingDraftAfterUserGesture = false;
                        }
                        const nowMs = Date.now();
                        const minTikTokActionGapMs = 900;
                        const actionSpacingElapsed = !tiktokState.lastActionAtMs
                            || (nowMs - Number(tiktokState.lastActionAtMs)) >= minTikTokActionGapMs;

                        if (captionText) {
                            const draftEditorContainer = bySelectors(['div.DraftEditor-editorContainer']);
                            const draftSpan = draftEditorContainer
                                ? draftEditorContainer.querySelector('span[data-text="true"]')
                                : null;
                            const draftEditable = draftSpan
                                ? draftSpan.closest('[contenteditable="true"]')
                                : (draftEditorContainer
                                    ? draftEditorContainer.querySelector('[contenteditable="true"]')
                                    : null);

                            const setTikTokCaption = (editableNode, spanNode, value) => {
                                if (!editableNode) return false;
                                const nextText = String(value || "");

                                const readCurrentText = () => {
                                    try {
                                        if (editableNode.isContentEditable || editableNode.getAttribute("contenteditable") === "true") {
                                            return normalizedNodeText(editableNode);
                                        }
                                        if ("value" in editableNode) {
                                            return norm(editableNode.value);
                                        }
                                    } catch (_) {}
                                    return "";
                                };

                                const replaceEditorTextWithoutExec = (textValue) => {
                                    try {
                                        editableNode.focus();
                                    } catch (_) {}

                                    try {
                                        if (editableNode.isContentEditable || editableNode.getAttribute("contenteditable") === "true") {
                                            const selection = window.getSelection();
                                            if (selection && typeof selection.removeAllRanges === "function") {
                                                const range = document.createRange();
                                                range.selectNodeContents(editableNode);
                                                selection.removeAllRanges();
                                                selection.addRange(range);
                                                range.deleteContents();

                                                if (textValue) {
                                                    const textNode = document.createTextNode(textValue);
                                                    range.insertNode(textNode);
                                                    range.setStartAfter(textNode);
                                                    range.collapse(true);
                                                    selection.removeAllRanges();
                                                    selection.addRange(range);
                                                }
                                                return true;
                                            }
                                        }
                                    } catch (_) {}

                                    try {
                                        if ("value" in editableNode) {
                                            editableNode.value = textValue;
                                            return true;
                                        }
                                    } catch (_) {}
                                    return false;
                                };

                                const eventOptions = { bubbles: true, composed: true };
                                try {
                                    editableNode.dispatchEvent(new InputEvent("beforeinput", { ...eventOptions, data: "", inputType: "deleteByCut" }));
                                } catch (_) {}

                                replaceEditorTextWithoutExec(nextText);

                                if (spanNode) {
                                    try { spanNode.textContent = nextText; } catch (_) {}
                                }

                                try {
                                    editableNode.dispatchEvent(new InputEvent("input", { ...eventOptions, data: nextText, inputType: "insertReplacementText" }));
                                } catch (_) {
                                    try { editableNode.dispatchEvent(new Event("input", eventOptions)); } catch (_) {}
                                }
                                try { editableNode.dispatchEvent(new Event("change", eventOptions)); } catch (_) {}
                                try { editableNode.dispatchEvent(new Event("blur", eventOptions)); } catch (_) {}

                                const current = readCurrentText();
                                return current === norm(nextText);
                            };

                            const captionAlreadyPresent = draftEditable
                                ? normalizedNodeText(draftEditable) === norm(captionText)
                                : false;
                            if (captionAlreadyPresent) {
                                textFilled = true;
                            } else if (actionSpacingElapsed && (draftEditable || draftSpan)) {
                                try {
                                    textFilled = setTikTokCaption(draftEditable, draftSpan, captionText) || textFilled;
                                    if (textFilled) {
                                        tiktokState.captionSetAtMs = Date.now();
                                        tiktokState.lastActionAtMs = tiktokState.captionSetAtMs;
                                    }
                                } catch (_) {}
                            }

                        }

                        captionReady = !captionText || textFilled;
                        const tiktokSubmitDelayElapsed = !captionText
                            || Boolean(tiktokState.captionSetAtMs && (Date.now() - Number(tiktokState.captionSetAtMs)) >= 1200);
                        const tiktokSubmitSpacingElapsed = !tiktokState.lastSubmitAttemptAtMs
                            || (Date.now() - Number(tiktokState.lastSubmitAttemptAtMs)) >= 1500;

                        const tiktokPostButton = bySelectors([
                            'button[data-e2e="save_draft_button"]',
                            'button[aria-disabled="false"][data-e2e="save_draft_button"]',
                        ]);
                        if (tiktokPostButton) {
                            const ariaDisabled = String(tiktokPostButton.getAttribute("aria-disabled") || "").toLowerCase();
                            const dataDisabled = String(tiktokPostButton.getAttribute("data-disabled") || "").toLowerCase();
                            const nativeDisabled = Boolean(tiktokPostButton.disabled);
                            tiktokPostEnabled = ariaDisabled === "false" && dataDisabled !== "true" && !nativeDisabled;
                            const waitingForDraftAfterGesture = Boolean(tiktokState.awaitingDraftAfterUserGesture);
                            const canSubmitNormally = captionReady && tiktokSubmitDelayElapsed && actionSpacingElapsed && tiktokSubmitSpacingElapsed;
                            const canSubmitAfterGesture = waitingForDraftAfterGesture && tiktokPostEnabled;
                            if (!tiktokState.submitClicked && tiktokPostEnabled && (canSubmitNormally || canSubmitAfterGesture)) {
                                submitClicked = clickNodeSingle(tiktokPostButton) || submitClicked;
                                if (submitClicked) {
                                    const clickedAtMs = Date.now();
                                    tiktokState.lastSubmitAttemptAtMs = clickedAtMs;
                                    tiktokState.lastActionAtMs = clickedAtMs;
                                    tiktokState.submitClicked = true;
                                    tiktokState.awaitingDraftAfterUserGesture = false;
                                }
                            }
                        }
                    }

                    if (platform === "youtube") {
                        const youtubeOptions = (payload.youtube_options && typeof payload.youtube_options === "object") ? payload.youtube_options : {};
                        const visibility = String(youtubeOptions.visibility || "public").toLowerCase();
                        const audience = String(youtubeOptions.audience || "not_kids").toLowerCase();
                        const nowMs = Date.now();
                        const actionSpacingElapsed = !youtubeState.lastActionAtMs || (nowMs - Number(youtubeState.lastActionAtMs)) >= 700;
                        const findYouTubeTextboxByLabel = (labelHint) => {
                            const hint = norm(labelHint);
                            const containers = collectDeep('ytcp-form-input-container, ytcp-mention-input, ytcp-social-suggestion-input');
                            for (const container of containers) {
                                let labelNode = null;
                                try {
                                    labelNode = container.querySelector('#label, [id="label"], label, .label, [aria-label]');
                                } catch (_) {}
                                const labelText = normalizedNodeText(labelNode || container);
                                if (!labelText.includes(hint)) continue;
                                let target = null;
                                try {
                                    target = container.querySelector('div#textbox[contenteditable="true"], textarea#textbox, #textbox[contenteditable="true"]');
                                } catch (_) {}
                                if (target) return target;
                            }
                            return null;
                        };

                        if (!fileInput && actionSpacingElapsed) {
                            const createButton = bySelectors([
                                'button[aria-label*="create" i]',
                                'ytcp-button#create-icon button',
                                'ytcp-button[id="create-icon"] button',
                                'ytcp-button#create-icon',
                            ]) || findClickableByHints(["create"]);
                            if (createButton) {
                                const clickedCreate = clickNodeOrAncestor(createButton);
                                openUploadClicked = clickedCreate || openUploadClicked;
                                if (clickedCreate) {
                                    youtubeState.lastActionAtMs = nowMs;
                                }
                            }

                            const uploadItem = bySelectors([
                                'tp-yt-paper-item[test-id="upload"]',
                                'tp-yt-paper-item#text-item-0',
                                'tp-yt-paper-item[aria-label*="upload" i]',
                                'ytd-menu-service-item-renderer tp-yt-paper-item[role="menuitem"]',
                            ]) || findClickableByHints(["upload videos", "upload video", "upload"]);
                            if (uploadItem) {
                                const clickedUpload = clickNodeOrAncestor(uploadItem);
                                openUploadClicked = clickedUpload || openUploadClicked;
                                if (clickedUpload) {
                                    youtubeState.uploadMenuClicked = true;
                                    youtubeState.lastActionAtMs = nowMs;
                                }
                            }
                        }

                        let youtubeTitleFilled = Boolean(youtubeState.titleFilled) || !titleText;
                        if (!youtubeTitleFilled && titleText) {
                            const titleTarget = findYouTubeTextboxByLabel("title") || bySelectors([
                                'textarea#textbox[aria-label*="title" i]',
                                'div#textbox[aria-label*="title" i][contenteditable="true"]',
                                'ytcp-mention-input[label*="Title" i] #textbox[contenteditable="true"]',
                                'ytcp-social-suggestion-input input#textbox',
                            ]);
                            youtubeTitleFilled = setTextValue(titleTarget, titleText) || youtubeTitleFilled;
                        }
                        youtubeState.titleFilled = youtubeTitleFilled;

                        let youtubeDescriptionFilled = Boolean(youtubeState.descriptionFilled) || !captionText;
                        if (!youtubeDescriptionFilled && captionText) {
                            const descTarget = findYouTubeTextboxByLabel("description") || bySelectors([
                                'ytcp-mention-input[label*="Description" i] #textbox[contenteditable="true"]',
                                'ytcp-social-suggestion-input #textbox[contenteditable="true"]',
                                'div#textbox[aria-label*="description" i][contenteditable="true"]',
                                'textarea#textbox[aria-label*="description" i]',
                            ]);
                            youtubeDescriptionFilled = setTextValue(descTarget, captionText) || youtubeDescriptionFilled;
                        }
                        youtubeState.descriptionFilled = youtubeDescriptionFilled;

                        textFilled = youtubeDescriptionFilled;

                        const madeForKidsLabel = findClickableByHints(["yes, it’s made for kids", "yes, it's made for kids"]);
                        const notKidsLabel = findClickableByHints(["no, it’s not made for kids", "no, it's not made for kids"]);
                        if (audience === "kids" && madeForKidsLabel) {
                            clickNodeOrAncestor(madeForKidsLabel);
                        } else if (audience !== "kids" && notKidsLabel) {
                            clickNodeOrAncestor(notKidsLabel);
                        }

                        const nextButton = bySelectors([
                            'ytcp-button#next-button button',
                            'ytcp-button[id="next-button"] button',
                            'button[aria-label*="next" i]',
                            'ytcp-uploads-dialog ytcp-button[id*="next" i] button',
                        ]) || findClickableByHints(["next"], { contexts: collectDeep('ytcp-uploads-dialog, tp-yt-paper-dialog, ytcp-dialog') });
                        const nextDisabled = Boolean(
                            nextButton
                            && (
                                nextButton.disabled
                                || String(nextButton.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
                                || String(nextButton.getAttribute('disabled') || '').toLowerCase() === 'true'
                            )
                        );
                        if (nextButton && !nextDisabled && Number(youtubeState.nextClicks || 0) < 3 && actionSpacingElapsed && youtubeTitleFilled && youtubeDescriptionFilled) {
                            const clicked = clickNodeOrAncestor(nextButton);
                            if (clicked) {
                                youtubeState.nextClicks = Number(youtubeState.nextClicks || 0) + 1;
                                youtubeState.lastActionAtMs = nowMs;
                                nextClicked = true;
                            }
                        }

                        const uploadDialog = bySelectors(['ytcp-uploads-dialog', 'tp-yt-paper-dialog'], { contexts: [document] });
                        const visibilityContexts = [uploadDialog, bySelectors(['ytcp-video-visibility-select'], { contexts: [uploadDialog || document] }), document].filter(Boolean);
                        const onVisibilityStep = Boolean(bySelectors([
                            'ytcp-video-visibility-select',
                            'tp-yt-paper-radio-group#privacy-radios',
                            'ytcp-button#done-button button',
                            'button[aria-label="Save"]',
                        ], { contexts: visibilityContexts }));

                        if (Number(youtubeState.nextClicks || 0) >= 3 || onVisibilityStep) {
                            const visibilityGroup = bySelectors([
                                'tp-yt-paper-radio-group#privacy-radios',
                                'ytcp-video-visibility-select tp-yt-paper-radio-group',
                            ], { contexts: visibilityContexts });
                            let visibilityRadio = visibility === "private"
                                ? bySelectors([
                                    'tp-yt-paper-radio-group#privacy-radios tp-yt-paper-radio-button#private-radio-button',
                                    'tp-yt-paper-radio-button#private-radio-button',
                                    'tp-yt-paper-radio-button[name="PRIVATE"]',
                                ], { contexts: visibilityContexts })
                                : (visibility === "unlisted"
                                    ? bySelectors([
                                        'tp-yt-paper-radio-group#privacy-radios tp-yt-paper-radio-button#unlisted-radio-button',
                                        'tp-yt-paper-radio-button#unlisted-radio-button',
                                        'tp-yt-paper-radio-button[name="UNLISTED"]',
                                    ], { contexts: visibilityContexts })
                                    : bySelectors([
                                        'tp-yt-paper-radio-group#privacy-radios tp-yt-paper-radio-button#public-radio-button',
                                        'tp-yt-paper-radio-button#public-radio-button',
                                        'tp-yt-paper-radio-button[name="PUBLIC"]',
                                    ], { contexts: visibilityContexts }));

                            if (!visibilityRadio && actionSpacingElapsed) {
                                const dialogScroller = bySelectors([
                                    'ytcp-uploads-dialog #scrollable-content',
                                    'ytcp-uploads-dialog #dialog',
                                    'ytcp-uploads-dialog',
                                ]);
                                try {
                                    if (dialogScroller && typeof dialogScroller.scrollBy === 'function') {
                                        dialogScroller.scrollBy({ top: 280, left: 0, behavior: 'instant' });
                                        youtubeState.lastActionAtMs = nowMs;
                                    }
                                } catch (_) {}
                            }

                            if (!visibilityRadio) {
                                const visibilityFallback = visibility === "private"
                                    ? findClickableByHints(["private"], { contexts: visibilityContexts })
                                    : (visibility === "unlisted"
                                        ? findClickableByHints(["unlisted"], { contexts: visibilityContexts })
                                        : findClickableByHints(["public"], { contexts: visibilityContexts }));
                                if (visibilityFallback && actionSpacingElapsed) {
                                    const clickedFallback = clickNodeOrAncestor(visibilityFallback);
                                    if (clickedFallback) {
                                        youtubeState.lastActionAtMs = nowMs;
                                    }
                                }
                                visibilityRadio = visibility === "private"
                                    ? bySelectors(['tp-yt-paper-radio-button#private-radio-button', 'tp-yt-paper-radio-button[name="PRIVATE"]'], { contexts: visibilityContexts })
                                    : (visibility === "unlisted"
                                        ? bySelectors(['tp-yt-paper-radio-button#unlisted-radio-button', 'tp-yt-paper-radio-button[name="UNLISTED"]'], { contexts: visibilityContexts })
                                        : bySelectors(['tp-yt-paper-radio-button#public-radio-button', 'tp-yt-paper-radio-button[name="PUBLIC"]'], { contexts: visibilityContexts }));
                            }

                            const visibilitySelected = Boolean(
                                visibilityRadio
                                && String(visibilityRadio.getAttribute('aria-checked') || '').toLowerCase() === 'true'
                            );
                            if (visibilityRadio && !visibilitySelected && actionSpacingElapsed) {
                                const innerRadioTarget = pick([
                                    visibilityRadio.querySelector && visibilityRadio.querySelector('div#radioContainer, div[role="radio"], #radioLabel, .onRadio, .offRadio'),
                                    visibilityRadio,
                                ]);
                                const clickedVisibility = clickNodeSingle(innerRadioTarget) || clickNodeOrAncestor(visibilityRadio);
                                if (clickedVisibility) {
                                    youtubeState.lastActionAtMs = nowMs;
                                }
                            }

                            const visibilityReady = Boolean(
                                visibilityRadio
                                && String(visibilityRadio.getAttribute('aria-checked') || '').toLowerCase() === 'true'
                            );

                            const scheduleText = String(youtubeOptions.schedule || "").trim();
                            if (scheduleText) {
                                const scheduleToggle = findClickableByHints(["schedule"]);
                                if (scheduleToggle) clickNodeOrAncestor(scheduleToggle);
                                const scheduleInput = bySelectors(['input[aria-label*="date" i]', 'input[placeholder*="date" i]']);
                                if (scheduleInput) {
                                    setTextValue(scheduleInput, scheduleText);
                                }
                            }

                            const finalActionHints = visibility === "public" ? ["save", "publish", "done"] : ["save", "done"];
                            const doneButton = bySelectors([
                                'ytcp-button#done-button button',
                                'ytcp-button[id="done-button"] button',
                                'button[aria-label="Save"]',
                                'button[aria-label*="save" i]',
                                visibility === "public" ? 'button[aria-label*="publish" i]' : 'button[aria-label*="done" i]',
                                'button[aria-label*="done" i]',
                            ], { contexts: visibilityContexts }) || findClickableByHints(finalActionHints, { contexts: visibilityContexts });
                            const doneDisabled = Boolean(
                                doneButton
                                && (
                                    doneButton.disabled
                                    || String(doneButton.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
                                    || String(doneButton.getAttribute('disabled') || '').toLowerCase() === 'true'
                                )
                            );
                            if (doneButton && !doneDisabled && !youtubeState.submitted && visibilityReady && actionSpacingElapsed) {
                                submitClicked = clickNodeOrAncestor(doneButton) || submitClicked;
                                if (submitClicked) {
                                    youtubeState.submitted = true;
                                    youtubeState.lastActionAtMs = nowMs;
                                }
                            }
                        }
                    }

                    return {
                        fileInputFound: Boolean(fileInput),
                        fileDialogTriggered,
                        openUploadClicked,
                        fileReadySignal,
                        textFilled,
                        captionReady,
                        facebookSubmitDelayElapsed,
                        nextClicked,
                        submitClicked,
                        tiktokPostEnabled,
                        tiktokSubmitClickedEver: Boolean(tiktokState.submitClicked),
                        videoPathQueued: Boolean(requestedVideoPath),
                        requestedVideoPath,
                        allowFileDialog,
                        fakeProbeInjected,
                    };
                } catch (err) {
                    return { error: String(err && err.stack ? err.stack : err) };
                }
            })()
        """.replace("__PAYLOAD_B64__", payload_b64)

        def _after(result):
            if platform_name not in self.social_upload_pending:
                return
            if isinstance(result, dict) and result.get("error"):
                self._append_log(f"WARNING: {platform_name} browser script error: {result.get('error')}")
                status_label.setText("Status: script error encountered; retrying...")
                timer.start(1700)
                return

            file_found = bool(isinstance(result, dict) and result.get("fileInputFound"))
            allow_file_dialog = bool(isinstance(result, dict) and result.get("allowFileDialog"))
            file_dialog_triggered = bool(isinstance(result, dict) and result.get("fileDialogTriggered"))
            open_upload_clicked = bool(isinstance(result, dict) and result.get("openUploadClicked"))
            file_ready_signal = bool(isinstance(result, dict) and result.get("fileReadySignal"))
            text_filled = bool(isinstance(result, dict) and result.get("textFilled"))
            next_clicked = bool(isinstance(result, dict) and result.get("nextClicked"))
            submit_clicked = bool(isinstance(result, dict) and result.get("submitClicked"))
            tiktok_post_enabled = bool(isinstance(result, dict) and result.get("tiktokPostEnabled"))
            tiktok_submit_clicked_ever = bool(isinstance(result, dict) and result.get("tiktokSubmitClickedEver"))
            video_path = str(self.social_upload_pending.get(platform_name, {}).get("video_path") or "").strip()
            video_path_exists = bool(video_path and Path(video_path).is_file())
            caption_queued = bool(str(self.social_upload_pending.get(platform_name, {}).get("caption") or "").strip())

            progress_bar.setVisible(True)
            progress_bar.setValue(min(95, 20 + attempts * 6))
            status_label.setText(
                f"Status: attempt {attempts} (file={'staged' if file_ready_signal else ('picker' if file_dialog_triggered else ('input' if file_found else 'no'))}, source={'ready' if video_path_exists else 'missing'}, caption={'manual' if caption_queued else 'none'})."
            )
            current_url = browser.url().toString().strip()
            tiktok_draft_landed = (
                platform_name == "TikTok"
                and "tiktokstudio/content" in current_url.lower()
                and "tab=draft" in current_url.lower()
            )
            self._append_log(
                f"{platform_name}: attempt {attempts} url={current_url or 'empty'} video_source={'set' if video_path_exists else 'missing'} allow_file_dialog={allow_file_dialog} results file_input={file_found} open_clicked={open_upload_clicked} file_picker={file_dialog_triggered} file_ready={file_ready_signal} caption_filled={text_filled} next_clicked={next_clicked} tiktok_post_enabled={tiktok_post_enabled} submit_clicked={submit_clicked}"
            )
            pending["allow_file_dialog"] = False

            is_tiktok = platform_name == "TikTok"
            is_youtube = platform_name == "YouTube"
            tiktok_upload_assumed = is_tiktok and attempts >= 3
            file_stage_ok = file_ready_signal or (file_found and file_dialog_triggered and video_path_exists) or tiktok_upload_assumed
            is_facebook = platform_name == "Facebook"
            is_instagram = platform_name == "Instagram"
            caption_ok = text_filled or not caption_queued
            submit_ok = (
                submit_clicked
                or (is_tiktok and (tiktok_submit_clicked_ever or tiktok_draft_landed))
            ) if (is_facebook or is_instagram or is_tiktok or is_youtube) else True
            completion_attempt_ready = submit_ok if (is_facebook or is_instagram or is_tiktok or is_youtube) else (attempts >= 2)
            if completion_attempt_ready and file_stage_ok and caption_ok and submit_ok:
                status_label.setText(
                    "Status: post submitted."
                    if (is_facebook or is_instagram or is_youtube)
                    else "Status: staged. Confirm/finalize post in this tab if needed."
                )
                progress_bar.setValue(100)
                self._append_log(
                    f"{platform_name} browser automation {'submitted post' if (is_facebook or is_instagram or is_youtube) else 'staged successfully'} in its tab."
                )
                self.social_upload_pending.pop(platform_name, None)
                return

            timer.start(1500)

        browser.page().runJavaScript(script, _after)

    def _start_upload(
        self,
        platform_name: str,
        upload_fn: Callable,
        upload_kwargs: dict,
        success_dialog_title: str,
        success_prefix: str,
    ) -> None:
        if self.upload_worker is not None and self.upload_worker.isRunning():
            QMessageBox.information(self, "Upload In Progress", "Please wait for the current upload to finish.")
            return

        self.upload_youtube_btn.setEnabled(False)

        self.upload_progress_label.setText(f"Upload progress: starting {platform_name} upload...")
        self.upload_progress_label.setVisible(True)
        self.upload_progress_bar.setValue(0)
        self.upload_progress_bar.setVisible(True)
        self._refresh_status_bar_visibility()
        self._append_log(f"Starting {platform_name} upload...")

        self.upload_worker = UploadWorker(platform_name=platform_name, upload_fn=upload_fn, upload_kwargs=upload_kwargs)
        self.upload_worker.progress.connect(self._on_upload_progress)
        self.upload_worker.finished_upload.connect(
            lambda platform, upload_id: self._on_upload_finished(platform, upload_id, success_dialog_title, success_prefix)
        )
        self.upload_worker.failed.connect(self._on_upload_failed)
        self.upload_worker.finished.connect(self._cleanup_upload_worker)
        self.upload_worker.start()

    def _on_upload_progress(self, value: int, message: str) -> None:
        bounded_value = max(0, min(100, int(value)))
        formatted_message = self._format_upload_progress_message(message)
        self.upload_progress_label.setText(f"Upload progress: {formatted_message}")
        self.upload_progress_label.setToolTip(str(message or ""))
        self.upload_progress_label.setVisible(bounded_value > 0)
        self.upload_progress_bar.setVisible(bounded_value > 0)
        self.upload_progress_bar.setValue(bounded_value)
        self._refresh_status_bar_visibility()

    def _format_upload_progress_message(self, message: str, max_length: int = 180) -> str:
        text = " ".join(str(message or "").split())
        if len(text) <= max_length:
            return text
        return text[: max_length - 1].rstrip() + "…"

    def _on_upload_finished(self, platform_name: str, upload_id: str, dialog_title: str, success_prefix: str) -> None:
        self.upload_progress_label.setText("Upload progress: complete")
        self.upload_progress_bar.setValue(100)
        self.upload_progress_bar.setVisible(False)
        self.upload_progress_label.setVisible(False)
        self._refresh_status_bar_visibility()
        self._append_log(f"{platform_name} upload complete. ID: {upload_id}")
        QMessageBox.information(self, dialog_title, f"{success_prefix} {upload_id}")

    def _on_upload_failed(self, platform_name: str, error_message: str) -> None:
        self.upload_progress_label.setText(f"Upload progress: failed ({error_message[:120]})")
        self.upload_progress_bar.setVisible(False)
        self.upload_progress_label.setVisible(False)
        self._refresh_status_bar_visibility()
        self._append_log(f"ERROR: {platform_name} upload failed: {error_message}")
        QMessageBox.critical(self, f"{platform_name} Upload Failed", error_message)

    def _cleanup_upload_worker(self) -> None:
        self.upload_youtube_btn.setEnabled(True)
        self.upload_worker = None

    def _compose_social_text(self, base_text: str, hashtags: list[str]) -> str:
        tag_text = " ".join(f"#{tag.lstrip('#')}" for tag in hashtags if tag.strip())
        if not tag_text:
            return base_text.strip()
        combined = f"{base_text.strip()}\n\n{tag_text}" if base_text.strip() else tag_text
        return combined.strip()

    def _build_tiktok_filename_stem(self, title_text: str, slogan_text: str, hashtags: list[str], max_length: int) -> str:
        safe_title = re.sub(r'[\\/:*?"<>|\r\n]+', " ", str(title_text or "")).strip()
        safe_slogan = re.sub(r'[\\/:*?"<>|\r\n]+', " ", str(slogan_text or "")).strip()
        safe_title = re.sub(r"\s+", " ", safe_title).strip(" .")
        safe_slogan = re.sub(r"\s+", " ", safe_slogan).strip(" .")
        normalized_tags = []
        for tag in hashtags:
            raw_tag = str(tag).strip().lstrip('#')
            if not raw_tag:
                continue
            clean_tag = re.sub(r'[\\/:*?"<>|\r\n]+', "", raw_tag)
            clean_tag = re.sub(r"\s+", "", clean_tag)
            clean_tag = clean_tag.strip(" .")
            if clean_tag:
                normalized_tags.append(f"#{clean_tag}")

        parts = [part for part in [safe_title, safe_slogan] if part]
        base = " - ".join(parts).strip()
        if not base:
            base = "Tiktok Upload"

        candidate_tags = normalized_tags.copy()
        if candidate_tags:
            stem = f"{base} {' '.join(candidate_tags)}".strip()
        else:
            stem = base

        while len(stem) > max_length and candidate_tags:
            candidate_tags.pop()
            stem = f"{base} {' '.join(candidate_tags)}".strip()

        if len(stem) > max_length:
            stem = stem[:max_length].rstrip(" .")
        return stem or "Tiktok Upload"

    def _stage_tiktok_browser_video(self, source_video_path: str, title_text: str, slogan_text: str, hashtags: list[str]) -> str:
        source_path = Path(str(source_video_path)).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise ValueError("TikTok upload video path is invalid.")

        extension = source_path.suffix or ".mp4"
        max_stem_length = max(1, 255 - len(extension))
        if os.name == "nt":
            max_windows_path = 240
            path_headroom = max_windows_path - len(str(source_path.parent)) - len(extension) - 1
            max_stem_length = max(16, min(max_stem_length, path_headroom))
        safe_stem = self._build_tiktok_filename_stem(title_text, slogan_text, hashtags, max_stem_length)
        staged_path = source_path.with_name(f"{safe_stem}{extension}")
        if staged_path == source_path:
            self._append_log(f"TikTok: using existing filename for browser upload: {source_path.name}")
            return str(source_path)

        counter = 1
        while staged_path.exists():
            staged_path = source_path.with_name(f"{safe_stem}-{counter}{extension}")
            counter += 1

        shutil.copy2(source_path, staged_path)
        self._append_log(
            f"TikTok: staged renamed upload file '{staged_path.name}' from description text before browser upload."
        )
        return str(staged_path)

    def _show_upload_dialog(self, platform_name: str, title_enabled: bool = True) -> tuple[str, str, list[str], str, bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{platform_name} Upload Details")
        dialog_layout = QVBoxLayout(dialog)

        title_input = QLineEdit()
        if title_enabled:
            dialog_layout.addWidget(QLabel(f"{platform_name} Title"))
            title_input.setText(self.ai_social_metadata.title)
            dialog_layout.addWidget(title_input)

        dialog_layout.addWidget(QLabel(f"{platform_name} Description / Caption"))
        description_input = QPlainTextEdit()
        description_input.setPlaceholderText("Describe this upload...")
        description_input.setPlainText(self.ai_social_metadata.description)
        dialog_layout.addWidget(description_input)

        dialog_layout.addWidget(QLabel("Hashtags (comma separated, no # needed)"))
        hashtags_input = QLineEdit(", ".join(self.ai_social_metadata.hashtags))
        dialog_layout.addWidget(hashtags_input)

        category_input = QLineEdit(self.ai_social_metadata.category)
        if platform_name == "YouTube":
            dialog_layout.addWidget(QLabel("YouTube Category ID"))
            dialog_layout.addWidget(category_input)

            visibility_input = QComboBox()
            visibility_input.addItem("Public", "public")
            visibility_input.addItem("Unlisted", "unlisted")
            visibility_input.addItem("Private", "private")
            dialog_layout.addWidget(QLabel("Visibility"))
            dialog_layout.addWidget(visibility_input)

            schedule_input = QLineEdit()
            schedule_input.setPlaceholderText("Optional schedule datetime (YYYY-MM-DD HH:MM)")
            dialog_layout.addWidget(QLabel("Schedule publish time"))
            dialog_layout.addWidget(schedule_input)

            audience_input = QComboBox()
            audience_input.addItem("Not made for kids", "not_kids")
            audience_input.addItem("Made for kids", "kids")
            dialog_layout.addWidget(QLabel("Audience"))
            dialog_layout.addWidget(audience_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        hashtags = [tag.strip().lstrip("#") for tag in hashtags_input.text().split(",") if tag.strip()]
        category_value = category_input.text().strip() if platform_name == "YouTube" else self.ai_social_metadata.category
        if platform_name == "YouTube":
            self.youtube_browser_upload_options = {
                "visibility": str(visibility_input.currentData() or "public"),
                "schedule": schedule_input.text().strip(),
                "audience": str(audience_input.currentData() or "not_kids"),
                "category": category_value,
            }
        return title_input.text().strip(), description_input.toPlainText().strip(), hashtags, category_value, accepted


if __name__ == "__main__":
    _configure_qtwebengine_runtime()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
