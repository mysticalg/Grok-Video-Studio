# Windows EXE/MSI Size Optimization Notes

If your packaged app is over 200 MB, the largest contributors are usually framework/runtime bundles rather than your Python source code.

## Biggest size drivers in this repo

1. **PySide6 + QtWebEngine**
   - QtWebEngine ships Chromium-based components and localization/resources.
   - This is typically the largest dependency in desktop Qt apps.
2. **Playwright runtime assets**
   - `playwright` Python package itself is moderate, but installed browser runtimes can be large.
3. **Bundled FFmpeg binaries** (if your installer includes them)
   - The app requires `ffmpeg` in PATH, but does not import the `ffmpeg` Python package.

## Cleanup already applied

- Removed obsolete/unneeded Python dependencies from the default install list:
  - Removed `google` (legacy meta-package, not imported by app code).
  - Removed `ffmpeg` (PyPI wrapper is not used; app calls system `ffmpeg` executable).

## Practical steps to reduce final installer size further

1. **Build one-file app without unnecessary optional features**
   - If you have a "lite" build, disable automation features that require Playwright.
2. **Do not bundle Playwright browser binaries in the MSI**
   - Keep browser install as post-install/setup step (`python -m playwright install chromium`) or an optional download.
3. **Trim Qt artifacts in your packager config**
   - Exclude unused Qt modules/plugins/translations if your build tool supports it.
   - Keep only platform plugin + image formats you use.
4. **Ship FFmpeg separately**
   - Keep FFmpeg as an external prerequisite (already how runtime is coded).
5. **Use compression in installer**
   - Enable maximum compression/LZMA in your MSI/installer tooling.

## Sanity checks before release

- Verify app startup and key tabs still work after exclusions.
- Verify YouTube upload flow (Google APIs) still works.
- Verify automation tab behavior when Playwright is present/absent.
- Verify stitch/interpolation/upscale still find `ffmpeg` in PATH.
