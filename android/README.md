# Android build and Play Store release

This Android project provides a production Android wrapper for **Grok Video Studio** and can generate installable APK/AAB artifacts for testing and Google Play submission.

## Feature parity model

The desktop app (`app.py`) is PySide-based and not directly executable on Android. To preserve the same end-user features on mobile, the Android app ships as a hardened WebView container and loads a full web deployment of Grok Video Studio.

- Configure the hosted full app URL with `APP_ENTRY_URL` in `android/app/build.gradle.kts`.
- You can also override it at launch time using an intent extra named `app_url`.
- Browser-based capabilities (generation, uploads, account login flows, content management) are available through the loaded web app.

## What this Android app includes

- Kotlin + Jetpack Compose full-screen WebView container
- JavaScript/DOM/media/cookie support for modern web app flows
- Native file upload picker (single and multi-select)
- DownloadManager integration for exported assets
- External deep-link handling (`mailto:`, app schemes, etc.)
- Release-safe build setup (R8 + resource shrinking)
- Optional env-based signing config for CI/CD

## Build requirements

- Android Studio Jellyfish+ or command-line Android SDK
- JDK 17
- Android SDK Platform 34 + Build Tools
- Gradle 8.14+

## Local build

From `android/`:

```bash
gradle assembleDebug
gradle bundleRelease
```

Artifacts:

- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`
- Release AAB: `app/build/outputs/bundle/release/app-release.aab`

## Configure release signing

Set these before `bundleRelease`:

- `ANDROID_KEYSTORE_PATH`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

Example:

```bash
export ANDROID_KEYSTORE_PATH=/path/to/release.jks
export ANDROID_KEYSTORE_PASSWORD='***'
export ANDROID_KEY_ALIAS='grokvideostudio'
export ANDROID_KEY_PASSWORD='***'
gradle bundleRelease
```

## Google Play release checklist

1. Create app in Play Console (`com.grokvideostudio.app`).
2. Upload `app-release.aab` to Internal testing.
3. Complete Data safety + privacy declarations.
4. Add store listing assets and screenshots.
5. Roll out staged production release.
