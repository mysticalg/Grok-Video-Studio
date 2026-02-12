# Grok Video to YouTube API (GUI)

A Flask GUI app that lets users:

1. Register/login (stored locally with hashed passwords)
2. Choose generation mode:
   - **Grok API mode** (chat + imagine endpoints)
   - **Web automation mode** (manual web login, then browser automation for Imagine prompt submission)
3. Batch-generate multiple 10-second videos (API mode includes 720p fallback to 420p)
4. Preview every generated video in-app
5. Download each video and upload each one individually to YouTube
6. Stitch selected videos together into one clip
7. Extract the last frame from any generated clip and use Grok vision context to generate a follow-up video

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
python app.py
```

Open: `http://localhost:5000`

## Web automation mode (manual login)

1. In dashboard settings, switch generation mode to **Web automation** and save.
2. Click **Start Manual Web Login Capture**.
3. A Chromium window opens; login manually and wait for Imagine prompt textbox to be visible.
4. Session is saved under `sessions/grok_web_<username>.json`.
5. Use **Generate Videos** with a manual prompt.

## Environment overrides

- `XAI_API_BASE` (default `https://api.x.ai/v1`)
- `GROK_CHAT_MODEL` (default `grok-3-mini`)
- `GROK_VIDEO_MODEL` (default `grok-video-latest`)
- `GROK_IMAGINE_URL` (default `https://grok.com/imagine`)
- `GROK_IMAGINE_PROMPT_SELECTOR` (default `textarea`)
- `GROK_IMAGINE_SUBMIT_SELECTOR` (default `button:has-text('Generate')`)
- `GROK_IMAGINE_VIDEO_SELECTOR` (default `video`)

## Notes

- The app stores user auth data in `users.db`.
- API keys/client secrets are stored in Flask session data for the active login session.
- YouTube OAuth token is persisted per user in `youtube_token_<username>.json`.
- Video files are downloaded under `downloads/`.
- Stitching and frame extraction require `ffmpeg` in your PATH.
