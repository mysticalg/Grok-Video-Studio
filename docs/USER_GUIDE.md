# Grok Video Studio - User Guide

This guide provides a full end-to-end workflow for setup, generation, post-processing, and social publishing.

## 1. Install and launch

1. Install Python 3.11+.
2. Install `ffmpeg` and ensure it is in your PATH.
3. Create a virtual environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (Linux/macOS) or `.\.venv\Scripts\Activate.ps1` (Windows)
   - `pip install -r requirements.txt`
4. Launch the app:
   - `python app.py`
5. Optional: install Playwright Chromium if you plan to use CDP automation:
   - `python -m playwright install chromium`

## 2. Configure credentials and models

Open **Model/API Settings** and configure:

- `GROK_API_KEY`
- `OPENAI_API_KEY` and/or `OPENAI_ACCESS_TOKEN` (as needed)
- `OLLAMA_API_BASE` and `OLLAMA_CHAT_MODEL` (optional)
- `SEEDANCE_API_KEY`
- Social upload credentials for YouTube, TikTok, Facebook, and Instagram

## 3. Generate videos

1. Enter concept and prompt controls in the left panel.
2. Select prompt generation source:
   - Grok API
   - OpenAI API
   - Ollama local model
3. Select video provider:
   - Grok Imagine API
   - OpenAI Sora 2 API
   - Seedance 2.0 API
4. Start generation and produce one or more variants.

## 4. Review, stitch, and export

1. Use **Generated Videos** to preview clips.
2. Select clips for stitching.
3. Enable optional pipeline features:
   - Crossfade transitions
   - 48/60fps interpolation
   - Upscale (2x/1080p/1440p/4K)
   - GPU encode
   - Music mix
4. Export final composition.

## 5. Publish to social platforms

Each upload tab supports API and browser automation paths.

### YouTube

1. Select source video.
2. Open **YouTube Upload** tab.
3. Choose **Upload via API** or **Automate in Browser**.
4. For browser automation, provide title, description, hashtags, audience, visibility, and optional schedule.

### TikTok / Facebook / Instagram

1. Select source video.
2. Open platform tab.
3. Upload by API or run browser automation.

## 6. Automation Chrome + CDP + UDP mode

1. Click **Start Automation Chrome**.
2. Click **Connect CDP**.
3. Optionally switch to **UDP** mode in Automation panel for workflow executor-based posting.
4. Run platform actions and monitor activity log.

## 7. Troubleshooting

- If stitching fails, verify `ffmpeg` installation.
- If API uploads fail, refresh credentials/tokens.
- If browser automation steps fail, platform UI selectors may have changed.
- Use Activity Log for diagnostics and execution traces.

## Workflow screenshots and PDF (local generation)

To avoid committing large/binary artifacts to Git hosting systems that reject binary diffs, generate the screenshots and PDF locally:

```bash
python docs/generate_user_manual.py
```

This produces:

- `docs/out/Grok-Video-Studio-User-Manual.pdf`
- `docs/out/assets/step-1-dashboard.png`
- `docs/out/assets/step-2-settings.png`
- `docs/out/assets/step-3-stitching.png`
- `docs/out/assets/step-4-upload.png`

Use these files for distribution releases or attachments outside the repo PR diff.
