import base64
import json
import os
import sqlite3
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from grok_web_automation import generate_video_via_web, manual_login_and_save
from youtube_uploader import upload_video_to_youtube

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"
DOWNLOAD_DIR = BASE_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
SESSION_DIR = BASE_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


API_BASE_URL = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")


def _api_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500] or response.reason

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err)
        if isinstance(err, str):
            return err
    return json.dumps(data)[:500]


def _extract_text_from_responses_api(payload: dict) -> str:
    output = payload.get("output", [])
    parts = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if text:
                    parts.append(text)
    if parts:
        return "\n".join(parts).strip()

    # fallback for schema changes
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    raise RuntimeError("No text output returned from responses API.")


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class GrokConfig:
    api_key: str
    chat_model: str
    image_model: str


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()


def get_user(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def create_user(username: str, password: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()


def logged_in() -> bool:
    return bool(session.get("username"))


def require_login():
    if not logged_in():
        flash("Please login first.", "error")
        return redirect(url_for("index"))
    return None


def load_grok_config() -> Optional[GrokConfig]:
    api_key = session.get("grok_api_key") or os.getenv("GROK_API_KEY")
    if not api_key:
        return None
    return GrokConfig(
        api_key=api_key,
        chat_model=session.get("chat_model", os.getenv("GROK_CHAT_MODEL", "grok-3-mini")),
        image_model=session.get("image_model", os.getenv("GROK_VIDEO_MODEL", "grok-video-latest")),
    )


def get_videos() -> list[dict]:
    return session.get("videos", [])


def save_videos(videos: list[dict]) -> None:
    session["videos"] = videos


def call_grok_chat(api_key: str, model: str, system: str, user: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Primary: OpenAI-compatible chat/completions endpoint
    chat_response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.9,
        },
        timeout=90,
    )
    if chat_response.ok:
        payload = chat_response.json()
        return payload["choices"][0]["message"]["content"].strip()

    # Fallback: responses API for tenants that reject chat/completions payload/schema
    responses_response = requests.post(
        f"{API_BASE_URL}/responses",
        headers=headers,
        json={
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "temperature": 0.9,
        },
        timeout=90,
    )
    if responses_response.ok:
        return _extract_text_from_responses_api(responses_response.json())

    message = (
        "Grok chat request failed. "
        f"chat/completions={chat_response.status_code}: {_api_error_message(chat_response)}; "
        f"responses={responses_response.status_code}: {_api_error_message(responses_response)}"
    )
    raise RuntimeError(message)


def call_grok_vision(api_key: str, model: str, image_path: Path, user_text: str) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    chat_response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.4,
        },
        timeout=120,
    )
    if chat_response.ok:
        return chat_response.json()["choices"][0]["message"]["content"].strip()

    responses_response = requests.post(
        f"{API_BASE_URL}/responses",
        headers=headers,
        json={
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"},
                    ],
                }
            ],
            "temperature": 0.4,
        },
        timeout=120,
    )
    if responses_response.ok:
        return _extract_text_from_responses_api(responses_response.json())

    raise RuntimeError(
        "Grok vision request failed. "
        f"chat/completions={chat_response.status_code}: {_api_error_message(chat_response)}; "
        f"responses={responses_response.status_code}: {_api_error_message(responses_response)}"
    )


def start_video_job(api_key: str, model: str, prompt: str, resolution: str, duration: int = 10) -> str:
    response = requests.post(
        f"{API_BASE_URL}/imagine/video/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,
            "duration_seconds": duration,
            "resolution": resolution,
            "fps": 24,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("id")


def poll_video_job(api_key: str, job_id: str, timeout_s: int = 420) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        response = requests.get(
            f"{API_BASE_URL}/imagine/video/generations/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
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
    raise TimeoutError("Timed out waiting for video generation.")


def download_video(video_url: str, username: str, suffix: str = "") -> Path:
    timestamp = int(time.time() * 1000)
    clean_suffix = f"_{suffix}" if suffix else ""
    file_path = DOWNLOAD_DIR / f"{username}_{timestamp}{clean_suffix}.mp4"
    with requests.get(video_url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return file_path


def generate_metadata(config: GrokConfig, concept: str, prompt: str) -> dict:
    metadata_raw = call_grok_chat(
        config.api_key,
        config.chat_model,
        (
            "Return ONLY compact JSON with keys: title, description, hashtags. "
            "hashtags must be an array of 5 short tags beginning with #."
        ),
        f"Create YouTube metadata for this concept and prompt. Concept: {concept}\nPrompt:{prompt}",
    )
    try:
        return _extract_json_object(metadata_raw)
    except json.JSONDecodeError:
        repaired = call_grok_chat(
            config.api_key,
            config.chat_model,
            "Convert the user text into strict compact JSON only.",
            (
                "Output JSON with keys title, description, hashtags where hashtags is an array of strings. "
                f"Text to convert:\n{metadata_raw}"
            ),
        )
        return _extract_json_object(repaired)


def generate_one_video(config: GrokConfig, concept: str, variant: int = 1, seed_context: str = "") -> dict:
    prompt = call_grok_chat(
        config.api_key,
        config.chat_model,
        "You write highly visual prompts for short cinematic AI videos.",
        (
            f"Create one polished video prompt for a 10 second scene in 720p from this concept: {concept}. "
            f"This is variant #{variant}. {seed_context}"
        ),
    )
    metadata = generate_metadata(config, concept, prompt)

    video_job_id = None
    chosen_resolution = None
    for resolution in ["1280x720", "640x420"]:
        try:
            video_job_id = start_video_job(config.api_key, config.image_model, prompt, resolution, duration=10)
            chosen_resolution = resolution
            break
        except requests.HTTPError:
            continue

    if not video_job_id:
        raise RuntimeError("Could not start a video generation job at 720p or 420p.")

    result = poll_video_job(config.api_key, video_job_id)
    video_url = result.get("output", {}).get("video_url") or result.get("video_url")
    if not video_url:
        raise RuntimeError("Generation succeeded but no video URL was returned.")

    file_path = download_video(video_url, session["username"], suffix=f"v{variant}")
    return {
        "id": str(int(time.time() * 1000)),
        "concept": concept,
        "prompt": prompt,
        "title": metadata.get("title", f"Grok Video {variant}"),
        "description": metadata.get("description", ""),
        "hashtags": " ".join(metadata.get("hashtags", [])),
        "resolution": chosen_resolution,
        "video_file_path": str(file_path),
        "youtube_video_id": "",
        "created_at": int(time.time()),
    }




def _web_session_state_path(username: str) -> Path:
    return SESSION_DIR / f"grok_web_{username}.json"


def generate_one_video_web(prompt: str, variant: int = 1) -> dict:
    username = session["username"]
    state_path = _web_session_state_path(username)
    output_path = DOWNLOAD_DIR / f"{username}_{int(time.time() * 1000)}_web_v{variant}.mp4"
    saved_file = generate_video_via_web(state_path, prompt=prompt, output_path=output_path)

    title = f"Web Imagine Clip {variant}"
    return {
        "id": str(int(time.time() * 1000)),
        "concept": prompt[:120],
        "prompt": prompt,
        "title": title,
        "description": "Generated via Grok web Imagine automation.",
        "hashtags": "#grok #imagine #ai",
        "resolution": "web",
        "video_file_path": str(saved_file),
        "youtube_video_id": "",
        "created_at": int(time.time()),
    }


def ensure_ffmpeg() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg is required for stitching and frame extraction.")


def stitch_video_paths(paths: list[str], output_path: Path) -> Path:
    ensure_ffmpeg()
    concat_file = output_path.with_suffix(".txt")
    concat_file.write_text("\n".join([f"file '{Path(p).resolve()}'" for p in paths]))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    concat_file.unlink(missing_ok=True)
    if run.returncode != 0:
        raise RuntimeError(run.stderr.strip() or "Video stitching failed")
    return output_path


def extract_last_frame(video_path: Path, frame_path: Path) -> Path:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.05",
        "-i",
        str(video_path),
        "-update",
        "1",
        "-q:v",
        "2",
        str(frame_path),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(run.stderr.strip() or "Could not extract last frame")
    return frame_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("index"))

        if action == "register":
            try:
                create_user(username, password)
                flash("Registration complete. Please log in.", "success")
            except sqlite3.IntegrityError:
                flash("Username already exists.", "error")
            return redirect(url_for("index"))

        user = get_user(username)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["username"] = username
            session["videos"] = []
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username/password.", "error")
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    if request.method == "POST":
        session["grok_api_key"] = request.form.get("grok_api_key", "").strip()
        session["chat_model"] = request.form.get("chat_model", os.getenv("GROK_CHAT_MODEL", "grok-3-mini")).strip()
        session["image_model"] = request.form.get("image_model", os.getenv("GROK_VIDEO_MODEL", "grok-video-latest")).strip()
        session["youtube_client_secret_json"] = request.form.get("youtube_client_secret_json", "").strip()
        session["generation_mode"] = request.form.get("generation_mode", "api").strip() or "api"
        flash("Settings saved to your session.", "success")
        return redirect(url_for("dashboard"))

    videos = get_videos()
    return render_template(
        "dashboard.html",
        username=session.get("username"),
        grok_api_key=session.get("grok_api_key", ""),
        chat_model=session.get("chat_model", os.getenv("GROK_CHAT_MODEL", "grok-3-mini")),
        image_model=session.get("image_model", os.getenv("GROK_VIDEO_MODEL", "grok-video-latest")),
        youtube_client_secret_json=session.get("youtube_client_secret_json", ""),
        generation_mode=session.get("generation_mode", "api"),
        videos=videos,
    )




@app.route("/web/login", methods=["POST"])
def web_login():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    timeout_s = max(60, min(900, int(request.form.get("timeout_s", "300") or "300")))
    try:
        state_path = _web_session_state_path(session["username"])
        manual_login_and_save(state_path, timeout_s=timeout_s)
        session["generation_mode"] = "web"
        flash("Web session captured. You can now generate with Web mode.", "success")
    except Exception as exc:
        flash(f"Web login failed: {exc}", "error")
    return redirect(url_for("dashboard"))


@app.route("/generate", methods=["POST"])
def generate():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    generation_mode = request.form.get("generation_mode") or session.get("generation_mode", "api")
    concept = request.form.get("concept", "").strip()
    manual_prompt = request.form.get("manual_prompt", "").strip()
    count = max(1, min(10, int(request.form.get("count", "1") or "1")))

    if generation_mode == "web":
        if not manual_prompt and not concept:
            flash("Add a prompt (or concept) for web Imagine generation.", "error")
            return redirect(url_for("dashboard"))
    else:
        config = load_grok_config()
        if not config:
            flash("Set Grok API key first in API settings or switch to Web mode.", "error")
            return redirect(url_for("dashboard"))
        if not concept:
            flash("Please add a concept.", "error")
            return redirect(url_for("dashboard"))

    videos = get_videos()
    created = 0
    for idx in range(1, count + 1):
        try:
            if generation_mode == "web":
                base_prompt = manual_prompt or concept
                variant_prompt = base_prompt if count == 1 else f"{base_prompt}\nVariation {idx} with distinct camera motion and composition."
                video = generate_one_video_web(variant_prompt, variant=idx)
            else:
                video = generate_one_video(config, concept, variant=idx)
            videos.insert(0, video)
            created += 1
        except Exception as exc:
            flash(f"Video {idx} failed: {exc}", "error")

    save_videos(videos)
    if created:
        flash(f"Successfully created {created} video(s) with mode={generation_mode}.", "success")
    return redirect(url_for("dashboard"))


@app.route("/video/<video_id>")
def stream_video(video_id: str):
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    for video in get_videos():
        if video["id"] == video_id and Path(video["video_file_path"]).exists():
            return send_file(video["video_file_path"], mimetype="video/mp4")
    flash("Video not found.", "error")
    return redirect(url_for("dashboard"))


@app.route("/download/<video_id>")
def download(video_id: str):
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    for video in get_videos():
        if video["id"] == video_id and Path(video["video_file_path"]).exists():
            return send_file(video["video_file_path"], as_attachment=True)
    flash("No generated video found.", "error")
    return redirect(url_for("dashboard"))


@app.route("/youtube/upload/<video_id>", methods=["POST"])
def youtube_upload(video_id: str):
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    videos = get_videos()
    target = next((v for v in videos if v["id"] == video_id), None)
    if not target or not Path(target["video_file_path"]).exists():
        flash("Video was not found.", "error")
        return redirect(url_for("dashboard"))

    client_secret_json = session.get("youtube_client_secret_json", "").strip()
    if not client_secret_json:
        flash("Paste your YouTube OAuth client secret JSON in API settings.", "error")
        return redirect(url_for("dashboard"))

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(client_secret_json)
            tmp_path = tmp.name

        title = request.form.get("title") or target["title"]
        description = request.form.get("description") or target["description"]
        hashtags = request.form.get("hashtags") or target["hashtags"]
        if hashtags:
            description = f"{description}\n\n{hashtags}".strip()

        token_file = str(BASE_DIR / f"youtube_token_{session['username']}.json")
        video_youtube_id = upload_video_to_youtube(
            client_secret_file=tmp_path,
            token_file=token_file,
            video_path=target["video_file_path"],
            title=title,
            description=description,
            tags=[tag.lstrip("#") for tag in hashtags.split() if tag.startswith("#")],
        )
        target["youtube_video_id"] = video_youtube_id
        save_videos(videos)
        flash(f"Uploaded successfully. YouTube Video ID: {video_youtube_id}", "success")
    except Exception as exc:
        flash(f"YouTube upload failed: {exc}", "error")
    finally:
        if "tmp_path" in locals() and Path(tmp_path).exists():
            Path(tmp_path).unlink()

    return redirect(url_for("dashboard"))


@app.route("/stitch", methods=["POST"])
def stitch():
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    selected_ids = request.form.getlist("video_ids")
    videos = get_videos()
    selected = [v for v in videos if v["id"] in selected_ids]
    if len(selected) < 2:
        flash("Pick at least 2 videos to stitch.", "error")
        return redirect(url_for("dashboard"))

    try:
        output = DOWNLOAD_DIR / f"{session['username']}_{int(time.time())}_stitched.mp4"
        stitched = stitch_video_paths([v["video_file_path"] for v in reversed(selected)], output)
        videos.insert(
            0,
            {
                "id": str(int(time.time() * 1000)),
                "concept": "stitched",
                "prompt": "Stitched from selected clips.",
                "title": "Stitched Grok Clips",
                "description": "Combined clips from batch generation.",
                "hashtags": "#grok #video",
                "resolution": "mixed",
                "video_file_path": str(stitched),
                "youtube_video_id": "",
                "created_at": int(time.time()),
            },
        )
        save_videos(videos)
        flash("Videos stitched successfully.", "success")
    except Exception as exc:
        flash(f"Stitch failed: {exc}", "error")

    return redirect(url_for("dashboard"))


@app.route("/generate/from-last-frame/<video_id>", methods=["POST"])
def generate_from_last_frame(video_id: str):
    auth_redirect = require_login()
    if auth_redirect:
        return auth_redirect

    config = load_grok_config()
    if not config:
        flash("Set Grok API key first in API settings.", "error")
        return redirect(url_for("dashboard"))

    videos = get_videos()
    base_video = next((v for v in videos if v["id"] == video_id), None)
    if not base_video or not Path(base_video["video_file_path"]).exists():
        flash("Source video not found.", "error")
        return redirect(url_for("dashboard"))

    try:
        frame_path = DOWNLOAD_DIR / f"{session['username']}_{int(time.time())}_last_frame.png"
        extract_last_frame(Path(base_video["video_file_path"]), frame_path)
        visual_context = call_grok_vision(
            config.api_key,
            config.chat_model,
            frame_path,
            "Describe this frame and propose a follow-up shot idea in one short paragraph.",
        )
        followup = generate_one_video(
            config,
            concept=f"Follow-up to prior generated clip. {base_video['concept']}",
            variant=1,
            seed_context=f"Use this visual continuity context: {visual_context}",
        )
        followup["prompt"] = f"{followup['prompt']}\n\n[Derived from last frame context: {visual_context}]"
        videos.insert(0, followup)
        save_videos(videos)
        flash("Generated follow-up video using the last-frame visual context.", "success")
    except Exception as exc:
        flash(f"Last-frame flow failed: {exc}", "error")

    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
