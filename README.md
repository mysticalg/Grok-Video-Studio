# Grok Video Studio

Desktop PySide6 app for generating AI videos, managing clips, previewing/stitching outputs, and publishing to social platforms with API + browser automation workflows.

This repo now also includes an Android module under `android/` for packaging an installable mobile app artifact (`.apk`/`.aab`) suitable for Google Play deployment.

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

## Android build (Google Play ready)

Use the Android project in `android/` to build and sign installable artifacts.

```bash
cd android
gradle assembleDebug
gradle bundleRelease
```

See full Android packaging/signing/deployment steps in [`android/README.md`](android/README.md).

Set the Android app entry URL (for full feature-parity web deployment) with `APP_ENTRY_URL` in `android/app/build.gradle.kts`.


## CDP relay architecture (QtWebEngine-safe automation)

To build a more durable automation layer that avoids brittle UI click scripts, this app can run QtWebEngine as the visual shell and expose Chromium DevTools Protocol (CDP) as the control plane:

- **Viewer:** embedded `QWebEngineView` tabs remain the UX surface for operators.
- **Controller:** a CDP relay can connect to the embedded browser target and drive behavior through:
  - DOM + Runtime introspection (state-aware actions)
  - Network event hooks (`Network.requestWillBeSent`, `Network.responseReceived`)
  - Request/response rewriting hooks where needed
- **Result:** fewer selector-only assumptions, better recovery from layout shifts, and a path to reusable automations across OpenAI Creator/Sora, TikTok, and YouTube flows.

### Enable QtWebEngine remote debugging

Set an environment variable before launch:

```bash
export GROK_QTWEBENGINE_REMOTE_DEBUG_PORT=9222
python app.py
```

The app will mirror this into `QTWEBENGINE_REMOTE_DEBUGGING` (unless already set), allowing an external CDP relay to attach to the embedded Chromium instance without changing the GUI workflow.

You can also enable this in **Model/API Settings → App Preferences**:
- Turn on **Enable QtWebEngine CDP remote debugging**
- Set **CDP Debug Port**
- Save settings and restart the app

For existing browser automations (TikTok/YouTube/Facebook/Instagram), you can optionally route each upload step through a local CDP relay in **Model/API Settings → App Preferences**:
- Enable **Use CDP relay for social browser automation**
- Set **CDP Relay URL** (default: `http://127.0.0.1:8765/social-upload-step`)
- If the relay is unavailable, the app automatically falls back to built-in DOM automation and pauses relay attempts for the current session (toggle relay mode off/on or restart to retry).


### Quickstart: run a local relay (no connection-refused errors)

If you do not have a relay service yet, start the included stub relay:

```bash
python tools/cdp_social_relay.py --host 127.0.0.1 --port 8765
```

Then in the app:
1. Open **Model/API Settings → App Preferences**
2. Enable **Use CDP relay for social browser automation**
3. Set **CDP Relay URL** to `http://127.0.0.1:8765/social-upload-step`
4. Save settings

What this relay does now:
- Connects to QtWebEngine via CDP (`QTWEBENGINE_REMOTE_DEBUGGING` port).
- Selects the active social page target and runs best-effort CDP DOM actions for TikTok/YouTube/Facebook/Instagram (caption/title/publish/share clicks).
- Returns `handled: true` when CDP step execution succeeds, with progress + status details.
- Returns `handled: false` automatically when CDP attach fails, so app fallback still works.

If CDP attach fails, verify remote debugging is enabled in App Preferences and restart the app after changing the debug port.
- On Windows, if a client drops the HTTP connection mid-response, the relay now treats it as non-fatal and continues serving subsequent requests.

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
