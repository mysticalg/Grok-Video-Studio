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

## Automation Chrome + CDP + Relay Extension architecture

The desktop app now supports a dedicated **Automation Chrome** controller path:

- Desktop launches real Chrome with `--remote-debugging-port=9222` and a dedicated user-data-dir.
- Desktop connects through Playwright `connect_over_cdp(...)` for browser-level control.
- A local WebSocket bus (`ws://127.0.0.1:18792`) enables extension ⇄ desktop commands/events.
- An unpacked MV3 extension (`extension/`) handles DOM checks/actions in the active tab via `chrome.scripting.executeScript`.
- QtWebEngine remains optional for UI only and is not used for CDP automation.

### Setup

After installing Python dependencies, install Playwright's Chromium runtime:

```bash
python -m playwright install chromium
```

### Runtime flow in app

1. Click **Start Automation Chrome** (starts WS bus + Chrome with relay extension loaded).
2. Click **Connect CDP** (desktop attaches to Chrome using the discovered `webSocketDebuggerUrl`).
3. Click **Extension DOM Ping** to send `dom.ping` and receive `cmd_ack` in the UI log.

### UDP automation mode (new backend)

Social browser posting now supports two coexisting executors:

- **Embedded**: existing QtWebEngine click automation.
- **UDP**: desktop workflow -> UDP command bus (`127.0.0.1:18793`) -> automation service -> relay extension DOM actions (+ CDP for page/tab and file-input handling).

In the app, use the automation mode selector in the **Automation Chrome + CDP** panel:

- `Embedded` keeps the prior behavior.
- `UDP` runs platform workflows through command actions (`platform.open`, `upload.select_file`, `form.fill`, `post.submit`, `post.status`) for YouTube, TikTok, Facebook, Instagram, and X tabs.

## Chrome extension ⇄ desktop app bridge (Native Messaging)

A starter implementation is included for a reliable extension-to-desktop control channel over Chrome Native Messaging (`stdin/stdout` JSON):

- Native host: `tools/chrome_native_host.py`
- Sample extension: `extension/native-bridge/`
- Setup guide: [`CHROME_NATIVE_MESSAGING.md`](CHROME_NATIVE_MESSAGING.md)

This is a good fit for automation control because Chrome supports it natively and the desktop app can keep privileged operations local.

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
