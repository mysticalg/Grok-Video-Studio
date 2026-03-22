# Grok Video Studio

[Join our Discord community](https://discord.gg/HRzMnuFB)

Desktop PySide6 app for generating AI videos, managing clips, previewing and stitching outputs, and publishing to social platforms with API and browser automation workflows.

## App screenshots

<p align="center">
  <img src="images/snap1.jpg" alt="Grok Video Studio main workspace with generation and upload controls" width="700" />
</p>

<p align="center">
  <img src="images/snap2.jpg" alt="Grok Video Studio workflow view showing video tooling and social publishing options" width="700" />
</p>

<p align="center">
  <em>Use the main workspace for setup and generation, then move into the workflow and publishing views for editing and uploads.</em>
</p>

## Quick start guide

Follow this path the first time you open the app:

1. **Install and launch**
   - Download the latest build from the GitHub Pages site below, or run from source with Python 3.11+ and `ffmpeg` installed.
   - Windows packaged builds include a bundled FFmpeg runtime for stitch, interpolate, upscale, and audio mix features.
   - Start the app with `python app.py` if you are running locally from source.
2. **Configure your keys and models**
   - Open **Model/API Settings**.
   - Add `GROK_API_KEY`, `OPENAI_API_KEY`, `SEEDANCE_API_KEY`, and optional Ollama settings if you plan to use them.
   - Save any social upload credentials you need before publishing.
3. **Generate clips**
   - Enter your concept and prompt controls in the left panel.
   - Choose a prompt source such as Grok, OpenAI, or Ollama.
   - Pick a video provider such as Grok Imagine, Sora 2, or Seedance, then generate one or more variants.
4. **Review, stitch, and export**
   - Use **Generated Videos** to preview outputs.
   - Stitch clips together and enable optional crossfade, interpolation, upscale, GPU encode, or music mix settings as needed.
   - Export the final composition.
5. **Publish to social platforms**
   - Select the final clip or exported file.
   - Open the upload tab for YouTube, TikTok, Facebook, or Instagram.
   - Choose **Upload via API** or **Automate in Browser**.
   - For browser automation, start **Automation Chrome**, connect CDP, and run the posting flow.

For the longer walkthrough with troubleshooting notes, open [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## GitHub Pages download site

- Live page: [https://mysticalg.github.io/Grok-Video-Studio/](https://mysticalg.github.io/Grok-Video-Studio/)
- Includes app screenshots, a step-by-step help section, and auto-updating links to the latest Windows MSI and EXE installers, macOS build, and Android APK from GitHub Releases.
- GitHub Actions auto-regenerates and deploys the downloads metadata on each published release via `.github/workflows/release-github-pages.yml`.

## Full documentation

- **User guide (Markdown):** [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- **Printable PDF manual (generated locally):** run `python docs/generate_user_manual.py` to create `docs/out/Grok-Video-Studio-User-Manual.pdf`
- **Workflow screenshots (generated locally):** created under `docs/out/assets/` by the same script
- **Windows build size notes:** [`docs/WINDOWS_BUILD_SIZE.md`](docs/WINDOWS_BUILD_SIZE.md)

## Latest feature highlights

- YouTube upload workflow includes browser automation for Studio metadata, audience and visibility, optional scheduling, and publish.
- Automation Chrome plus CDP supports extension relay messaging and DOM command execution.
- UDP automation mode is available for cross-platform browser posting workflows across YouTube, TikTok, Facebook, Instagram, and X.
- Stitch pipeline includes crossfade, interpolation, upscale presets, GPU encode options, and music mixing.
- AI Flow Trainer is available for recording and replaying browser workflows.

## Features

- **Video generation**
  - Manual prompt workflow in embedded `grok.com/imagine` browser
  - Prompt generation via xAI Grok API, OpenAI API, or local Ollama API
  - Video providers: Grok Imagine API, OpenAI Sora 2 API, Seedance 2.0 API
  - Batch and variant queue execution
  - Continue-from-last-frame and continue-from-local-image tools
- **Media pipeline**
  - Generated Videos picker with thumbnail previews
  - In-app video playback controls for seek, volume, mute, and fullscreen
  - Stitch clips with optional crossfade, interpolation (48 or 60 fps), upscale (2x, 1080p, 1440p, or 4K), GPU encode, and music mixing
- **Social publishing**
  - Dedicated upload tabs for Facebook, Instagram, TikTok, and YouTube
  - Per-platform API upload actions
  - Browser automation uploader for each platform
  - YouTube upload tab includes browser automation to open Studio, upload a file, set title and description, pick audience and visibility, optionally schedule, and publish
- **Automation**
  - AI Flow Trainer tab to record, build, and replay browser workflows
- **UX updates**
  - Buy Me a Coffee button moved under Activity Log and made smaller
  - Generated Videos area is taller for easier selection

## Supported models

- **Grok chat and prompt models** (configurable), default: `grok-3-mini`
- **Grok video model** (configurable), default: `grok-video-latest`
- **OpenAI chat model** (configurable), default: `gpt-5.1-codex`
- **Ollama chat model** (configurable), default: `llama3.1:8b` on `http://127.0.0.1:11434/v1`
- **OpenAI Sora models** (Sora 2 tab examples include `sora-2`, `sora-2-pro`, and dated variants)
- **Seedance model settings** via Seedance tab

## Install

### 1. Prerequisites

- Python 3.11+ (native OS Python is fine, including 3.14)
- `ffmpeg` in PATH (required for stitch, interpolate, upscale, and audio mix)
  - Windows packaged releases bundle FFmpeg automatically; source runs still expect `ffmpeg` in PATH.

### 2. Create env and install deps

Use the launcher that maps to your system Python install: `python`, `python3`, or `py`.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```

## Automation Chrome + CDP + Relay Extension architecture

The desktop app supports a dedicated **Automation Chrome** controller path:

- Desktop launches real Chrome with `--remote-debugging-port=9222` and a dedicated user-data-dir.
- Desktop connects through Playwright `connect_over_cdp(...)` for browser-level control.
- A local WebSocket bus (`ws://127.0.0.1:18792`) enables extension <-> desktop commands and events.
- An unpacked MV3 extension (`extension/`) handles DOM checks and actions in the active tab via `chrome.scripting.executeScript`.
- QtWebEngine remains optional for UI only and is not used for CDP automation.

### Setup

After installing Python dependencies, install Playwright's Chromium runtime:

```bash
python -m playwright install chromium
```

### Runtime flow in app

1. Click **Start Automation Chrome** to start the WebSocket bus and Chrome with the relay extension loaded.
2. Click **Connect CDP** to attach the desktop app to Chrome using the discovered `webSocketDebuggerUrl`.
3. Click **Extension DOM Ping** to send `dom.ping` and receive `cmd_ack` in the UI log.

### UDP automation mode

Social browser posting supports two coexisting executors:

- **Embedded:** existing QtWebEngine click automation
- **UDP:** desktop workflow -> UDP command bus (`127.0.0.1:18793`) -> automation service -> relay extension DOM actions plus CDP for page, tab, and file-input handling

In the app, use the automation mode selector in the **Automation Chrome + CDP** panel:

- `Embedded` keeps the prior behavior.
- `UDP` runs platform workflows through command actions such as `platform.open`, `upload.select_file`, `form.fill`, `post.submit`, and `post.status` for YouTube, TikTok, Facebook, Instagram, and X tabs.

## Chrome extension <-> desktop app bridge (Native Messaging)

A starter implementation is included for a reliable extension-to-desktop control channel over Chrome Native Messaging (`stdin/stdout` JSON):

- Native host: `tools/chrome_native_host.py`
- Sample extension: `extension/native-bridge/`
- Setup guide: [`CHROME_NATIVE_MESSAGING.md`](CHROME_NATIVE_MESSAGING.md)

This is a good fit for automation control because Chrome supports it natively and the desktop app can keep privileged operations local.

## Android (Google Play)

An Android project is included at `android/` so CI can generate both:

- a Play-ready `.aab` bundle for Google Play
- a release `.apk` for direct installs and testing

The Android app is now a native Kotlin/Compose client rather than a webpage container. Current Android-native flows include:

- prompt drafting and prompt or metadata generation
- direct video generation requests for Grok Imagine, OpenAI Sora, and Seedance
- on-device video library with playback and import
- basic on-device stitch export for saved clips
- mobile publish draft editing plus Android share-target hand-off

Still desktop-only for now:

- browser automation uploaders
- advanced stitch and export controls like crossfade, interpolation, upscale, and music mixing
- AI Flow Trainer and the larger desktop browser and tooling stack

### Build locally

```bash
gradle -p android assembleDebug
gradle -p android bundleRelease assembleRelease
```

Release outputs:

- `android/app/build/outputs/apk/debug/app-debug.apk`
- `android/app/build/outputs/bundle/release/app-release.aab`
- `android/app/build/outputs/apk/release/app-release.apk`
- `android/app/build/outputs/apk/release/app-release-unsigned.apk` when signing is not configured

### Versioning

- `versionName` defaults to the repository `VERSION` file.
- `versionCode` is derived from `VERSION` for local builds.
- GitHub Actions overrides both values for CI releases so Play uploads always get a monotonically increasing `versionCode`.

### Optional release signing

For local Gradle signing, create `android/release-keystore.properties`:

```properties
storeFile=release.keystore
storePassword=your_store_password
keyAlias=your_key_alias
keyPassword=your_key_password
```

In GitHub Actions, the workflow decodes the keystore and writes `android/release-keystore.properties` before running the Android release build when these repository secrets are set:

- `ANDROID_SIGNING_KEY_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

### Optional Google Play publishing

When `GOOGLE_PLAY_SERVICE_ACCOUNT_JSON` is configured, the Android release workflow can upload the signed `.aab` to Google Play automatically.

Recommended repository configuration:

- Secret: `GOOGLE_PLAY_SERVICE_ACCOUNT_JSON`
- Variable: `GOOGLE_PLAY_PACKAGE_NAME` (defaults to `com.grokvideostudio.app`)
- Variable: `GOOGLE_PLAY_TRACK` (defaults to `internal`)
- Variable: `GOOGLE_PLAY_RELEASE_STATUS` (defaults to `completed`)

Notes:

- Google Play publishing requires the Android signing secrets above.
- The package must already exist in Play Console, so the first manual upload and setup still needs to be done once.
- The workflow still uploads the `.apk` and `.aab` as GitHub artifacts and release assets even when Play publishing is not configured.

The CI build also validates the final `.aab` with `bundletool validate` and fails fast if `BundleConfig.pb` is missing.

## Configure credentials

In the **Model/API Settings** tab configure what you need:

- `GROK_API_KEY`
- `OPENAI_API_KEY` and/or `OPENAI_ACCESS_TOKEN`
- `OLLAMA_API_BASE` and `OLLAMA_CHAT_MODEL` (optional, for local prompt and caption generation)
- `SEEDANCE_API_KEY` (or OAuth token)
- Upload credentials for YouTube, Facebook, Instagram, and TikTok

You can set env vars first, or enter them directly in the UI.

## Interface guide

### Left panel

1. Enter concept and prompt settings.
2. Choose a prompt source and video provider.
3. Run generate actions.
4. Manage clips in the **Generated Videos** list.
5. Use stitch and export tools.

### Right panel tabs

- **Browser:** embedded Grok imagine flow
- **Facebook Upload / Instagram Upload / TikTok Upload / YouTube Upload**
  - `Upload via API`
  - `Automate in Browser`
  - `Open Upload Page`
- **Sora 2 Video Settings:** OpenAI video params
- **Seedance 2.0 Video Settings:** Seedance params
- **AI Flow Trainer:** train, build, and run automation

### YouTube browser automation workflow

1. Select a generated or local video.
2. Open the **YouTube Upload** tab.
3. Click **Automate YouTube in Browser**.
4. Fill in title, description and hashtags, category, visibility, audience, and optional schedule datetime.
5. The app opens or reuses a YouTube Studio tab, uploads the file, fills metadata, steps through the Studio flow, and publishes.

> Note: YouTube UI can change. If selectors drift, automation may require manual final confirmation in the tab.

## Upload notes

- **YouTube API upload** uses `youtube_token.json` (or `client_secret.json` for OAuth bootstrap).
- **Facebook, Instagram, and TikTok** include both API and browser automation paths.
- TikTok and Facebook OAuth helper flows are available in settings.

## Legal

- [Terms of Service](TERMS_OF_SERVICE.md)
- [Privacy Policy](PRIVACY_POLICY.md)

## Support

- Buy Me a Coffee: https://buymeacoffee.com/dhooksterm
- PayPal: https://www.paypal.com/paypalme/dhookster
- SOL: `6HiqW3jeF3ymxjK5Fcm6dHi46gDuFmeCeSNdW99CfJjp`
