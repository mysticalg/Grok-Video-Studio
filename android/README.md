# Android build and Play Store release

This Android module wraps Grok Video Studio into a mobile-first app shell and is ready to produce installable APK/AAB artifacts for internal testing and Google Play deployment.

## What this Android port contains

- Kotlin + Jetpack Compose launcher app
- Full-screen `WebView` shell pointed at `https://grok.com/imagine`
- File picker support for uploading videos/images from Android storage
- Production release config with R8 shrinking/minification
- Environment-driven signing config for CI/CD

## Build requirements

- Android Studio Jellyfish+ or command-line Android SDK
- JDK 17
- Android SDK Platform 34 + Build Tools
- Gradle (or Android Studio managed Gradle)

## Local build

From the `android/` directory:

```bash
./gradlew assembleDebug
./gradlew bundleRelease
```

Generated artifacts:

- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`
- Release AAB: `app/build/outputs/bundle/release/app-release.aab`

## Configure release signing

Set these environment variables before building release artifacts:

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
./gradlew bundleRelease
```

## Google Play upload checklist

1. Create app in Play Console (`com.grokvideostudio.app`).
2. Upload `app-release.aab` to internal testing track first.
3. Complete Data safety + privacy disclosures.
4. Upload store listing screenshots, icon, feature graphic.
5. Promote to production after test rollout.
