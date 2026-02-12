# Grok Video Desktop Studio (Windows GUI)

This project is now a **desktop Python GUI app** instead of a Flask web server.

## What changed

- No web server routes/pages.
- Uses a split-pane desktop layout.
- The **browser is permanently embedded in the right-hand pane** using Qt WebEngine.
- The left pane contains Grok settings, concept input, generate controls, and logs.

## Features

1. Enter Grok API key, chat model, and video model.
2. Generate one or more video variants from a concept.
3. Keep a generated-video list in the GUI session.
4. Select any generated item to load/play it in the embedded browser pane.
5. Open a local video file and view it in the same embedded browser pane.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Environment variables

- `GROK_API_KEY`
- `GROK_CHAT_MODEL` (default: `grok-3-mini`)
- `GROK_VIDEO_MODEL` (default: `grok-video-latest`)
- `XAI_API_BASE` (default: `https://api.x.ai/v1`)

## Notes

- Downloaded videos are saved under `downloads/`.
- The right-hand pane is always present and used for embedded browsing/video preview.
