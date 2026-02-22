# Chrome Extension ⇄ Desktop App (Native Messaging)

Yes — this repo can support a reliable **Chrome extension ⇄ desktop app** channel using Chrome Native Messaging (`stdin/stdout` JSON frames).

## Included scaffold

- Native host script: `tools/chrome_native_host.py`
- Example MV3 extension: `extension/native-bridge/`

## 1) Native host manifest

Create a native host manifest named `com.grok.video_studio.host.json`:

```json
{
  "name": "com.grok.video_studio.host",
  "description": "Grok Video Studio native messaging host",
  "path": "/ABSOLUTE/PATH/TO/python",
  "type": "stdio",
  "allowed_origins": [
    "chrome-extension://<YOUR_EXTENSION_ID>/"
  ]
}
```

On Linux, the host command is generally a tiny launcher script, for example:

```bash
#!/usr/bin/env bash
exec /usr/bin/python3 /workspace/Grok-Video-Studio/tools/chrome_native_host.py
```

Set that launcher path in `path`.

## 2) Register manifest with Chrome

Typical locations:

- Linux user: `~/.config/google-chrome/NativeMessagingHosts/com.grok.video_studio.host.json`
- macOS user: `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.grok.video_studio.host.json`
- Windows user: registry key under
  `HKCU\Software\Google\Chrome\NativeMessagingHosts\com.grok.video_studio.host`
  with default value pointing to manifest file path.

## 3) Load extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select `extension/native-bridge`
4. Copy extension ID and update `allowed_origins` in the native host manifest.

## 4) Message contract

Requests (extension → host):

- `{"id":"1","action":"ping"}`
- `{"id":"2","action":"social_upload_step","payload":{...},"relay_url":"http://127.0.0.1:8765/social-upload-step"}`
- `{"id":"3","action":"http_proxy","url":"http://127.0.0.1:8765/health","method":"POST","body":{}}`

Response shape:

```json
{
  "id": "2",
  "ok": true,
  "action": "social_upload_step",
  "result": {"handled": true}
}
```

## Why this architecture works well

- Chrome Native Messaging is stable and officially supported.
- Desktop app keeps privileged operations local.
- Extension can stay minimal while desktop side performs automation and relay calls.
