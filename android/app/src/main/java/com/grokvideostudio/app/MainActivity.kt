package com.grokvideostudio.app

import android.annotation.SuppressLint
import android.net.Uri
import android.os.Bundle
import android.webkit.CookieManager
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.webkit.WebSettingsCompat
import androidx.webkit.WebViewFeature

private const val START_URL = "https://grok.com/imagine"

class MainActivity : ComponentActivity() {
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        fileUploadCallback?.onReceiveValue(uris.toTypedArray())
        fileUploadCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    GrokVideoStudioWebApp(
                        launchFilePicker = { callback ->
                            fileUploadCallback?.onReceiveValue(null)
                            fileUploadCallback = callback
                            filePickerLauncher.launch(arrayOf("video/*", "image/*"))
                        }
                    )
                }
            }
        }
    }

    @Composable
    private fun GrokVideoStudioWebApp(
        launchFilePicker: (ValueCallback<Array<Uri>>) -> Unit,
    ) {
        var isLoading by remember { mutableStateOf(true) }
        var webView by remember { mutableStateOf<WebView?>(null) }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                modifier = Modifier.fillMaxSize(),
                factory = { context ->
                    WebView(context).apply {
                        webView = this
                        CookieManager.getInstance().setAcceptCookie(true)
                        CookieManager.getInstance().setAcceptThirdPartyCookies(this, true)

                        settings.javaScriptEnabled = true
                        settings.domStorageEnabled = true
                        settings.loadsImagesAutomatically = true
                        settings.mediaPlaybackRequiresUserGesture = false
                        settings.cacheMode = WebSettings.LOAD_DEFAULT
                        settings.allowFileAccess = true
                        settings.allowContentAccess = true
                        settings.mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
                        settings.userAgentString = settings.userAgentString + " GrokVideoStudioAndroid/1.0"

                        if (WebViewFeature.isFeatureSupported(WebViewFeature.ALGORITHMIC_DARKENING)) {
                            WebSettingsCompat.setAlgorithmicDarkeningAllowed(settings, true)
                        }

                        webViewClient = object : WebViewClient() {
                            override fun shouldOverrideUrlLoading(
                                view: WebView?,
                                request: WebResourceRequest?
                            ): Boolean {
                                return false
                            }

                            override fun onPageFinished(view: WebView?, url: String?) {
                                isLoading = false
                            }
                        }

                        webChromeClient = object : WebChromeClient() {
                            override fun onShowFileChooser(
                                webView: WebView?,
                                filePathCallback: ValueCallback<Array<Uri>>?,
                                fileChooserParams: FileChooserParams?
                            ): Boolean {
                                if (filePathCallback == null) return false
                                launchFilePicker(filePathCallback)
                                return true
                            }
                        }

                        loadUrl(START_URL)
                    }
                },
                update = {
                    webView = it
                }
            )

            if (isLoading) {
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            }
        }

        LaunchedEffect(webView) {
            webView?.resumeTimers()
            webView?.onResume()
        }

        DisposableEffect(webView) {
            onDispose {
                webView?.onPause()
                webView?.pauseTimers()
                webView?.destroy()
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        // Allow in-webview back navigation.
        val rootView = window.decorView.rootView
        val webView = findWebView(rootView)
        if (webView != null && webView.canGoBack()) {
            webView.goBack()
            return
        }
        super.onBackPressed()
    }

    private fun findWebView(view: android.view.View): WebView? {
        if (view is WebView) return view
        if (view is android.view.ViewGroup) {
            for (i in 0 until view.childCount) {
                val found = findWebView(view.getChildAt(i))
                if (found != null) return found
            }
        }
        return null
    }
}
