# Grok Video Studio

Desktop PySide6 app for generating AI videos, managing clips, previewing/stitching outputs, and publishing to social platforms with API + browser automation workflows.

## Features

- **Video generation**
  - Manual prompt workflow in embedded `grok.com/imagine` browser
  - Prompt generation via **xAI Grok API** or **OpenAI API**
  - Video providers: **Grok Imagine API**, **OpenAI Sora 2 API**, **Seedance 2.0 API**
  - Batch/variant queue execution
  - Continue-from-last-frame and continue-from-local-image tools
- **Media pipeline**
  - Generated Videos picker with thumbnail previews
  - In-app video playback controls (seek/volume/mute/fullscreen)
  - Stitch clips with optional crossfade, interpolation (48/60fps), upscale (2x/1080p/1440p/4K), GPU encode, and music mixing
- **Social publishing**
  - Dedicated upload tabs: **Facebook**, **Instagram**, **TikTok**, **YouTube**
  - Per-platform API upload actions
  - Browser automation uploader for each platform
  - New YouTube upload tab includes browser automation to load Studio, upload file, set title/description, pick audience/visibility, optional scheduling, and publish
- **Automation**
  - AI Flow Trainer tab to record/build/replay browser workflows
- **UX updates**
  - Buy Me a Coffee button moved under Activity Log and made smaller
  - Generated Videos area is taller for easier selection

## Supported models

- **Grok chat/prompt models** (configurable), default: `grok-3-mini`
- **Grok video model** (configurable), default: `grok-video-latest`
- **OpenAI chat model** (configurable), default: `gpt-5.1-codex`
- **OpenAI Sora models** (Sora 2 tab examples include `sora-2`, `sora-2-pro`, dated variants)
- **Seedance model settings** via Seedance tab

## Install

### 1) Prerequisites

- Python 3.11+
- `ffmpeg` in PATH (required for stitch/interpolate/upscale/audio mix)

### 2) Create env + install deps

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Run

```bash
python app.py
```

## Configure credentials

In **Model/API Settings** tab configure what you need:

- `GROK_API_KEY`
- `OPENAI_API_KEY` and/or `OPENAI_ACCESS_TOKEN`
- `SEEDANCE_API_KEY` (or OAuth token)
- Upload credentials (YouTube/Facebook/Instagram/TikTok)

You can set env vars first, or enter directly in the UI.

## Interface guide

### Left panel

1. Enter concept/prompt settings.
2. Choose prompt source and video provider.
3. Run generate actions.
4. Manage clips in **Generated Videos** list.
5. Use stitch/export tools.

### Right panel tabs

- **Browser**: embedded Grok imagine flow
- **Facebook Upload / Instagram Upload / TikTok Upload / YouTube Upload**:
  - `Upload via API`
  - `Automate in Browser`
  - `Open Upload Page`
- **Sora 2 Video Settings**: OpenAI video params
- **Seedance 2.0 Video Settings**: Seedance params
- **AI Flow Trainer**: train/build/run automation

### YouTube browser automation workflow

1. Select a generated/local video.
2. Open **YouTube Upload** tab.
3. Click **Automate YouTube in Browser**.
4. Fill dialog values:
   - title
   - description + hashtags
   - category
   - visibility (public/unlisted/private)
   - audience
   - optional schedule datetime
5. App opens/uses YouTube Studio tab, uploads file, fills metadata, steps through Studio flow, and publishes.

> Note: YouTube UI can change. If selectors drift, automation may require manual final confirmation in-tab.

## Upload notes

- **YouTube API upload** uses `youtube_token.json` (or `client_secret.json` for OAuth bootstrap).
- **Facebook/Instagram/TikTok** include both API and browser automation paths.
- TikTok/Facebook OAuth helper flows are available in settings.

## Legal

- [Terms of Service](TERMS_OF_SERVICE.md)
- [Privacy Policy](PRIVACY_POLICY.md)

## Support

- Buy Me a Coffee: https://buymeacoffee.com/dhooksterm
- PayPal: https://www.paypal.com/paypalme/dhookster
- SOL: `6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp`
