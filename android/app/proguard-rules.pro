# Keep WebView Javascript interfaces if added in future.
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}
