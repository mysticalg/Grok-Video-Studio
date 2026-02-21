package com.grokvideostudio.app

import android.annotation.SuppressLint
import android.app.DownloadManager
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.webkit.CookieManager
import android.webkit.DownloadListener
import android.webkit.PermissionRequest
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.OpenMultipleDocuments
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView


class MainActivity : ComponentActivity() {
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null

    private val openDocumentsLauncher = registerForActivityResult(OpenMultipleDocuments()) { uris ->
        fileUploadCallback?.onReceiveValue(uris.toTypedArray())
        fileUploadCallback = null
    }

    private val openSingleDocumentLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        fileUploadCallback?.onReceiveValue(uri?.let { arrayOf(it) })
        fileUploadCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val appUrl = intent?.getStringExtra("app_url")?.takeIf { it.isNotBlank() } ?: BuildConfig.APP_ENTRY_URL

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    GrokVideoStudioWebApp(
                        appUrl = appUrl,
                        launchFilePicker = { callback, allowMultiple ->
                            fileUploadCallback?.onReceiveValue(null)
                            fileUploadCallback = callback
                            if (allowMultiple) {
                                openDocumentsLauncher.launch(arrayOf("video/*", "image/*", "*/*"))
                            } else {
                                openSingleDocumentLauncher.launch(arrayOf("video/*", "image/*", "*/*"))
                            }
                        },
                        onExternalUrlRequested = ::openExternalLink,
                        onDownloadRequested = ::enqueueDownload,
                    )
                }
            }
        }
    }

    private fun openExternalLink(uri: Uri) {
        val browserIntent = Intent(Intent.ACTION_VIEW, uri)
        try {
            startActivity(browserIntent)
        } catch (_: ActivityNotFoundException) {
            Toast.makeText(this, "No app found to open this link.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun enqueueDownload(
        url: String,
        userAgent: String?,
        contentDisposition: String?,
        mimeType: String?,
    ) {
        val fileName = URLUtil.guessFileName(url, contentDisposition, mimeType)
        val request = DownloadManager.Request(Uri.parse(url)).apply {
            setTitle(fileName)
            setDescription("Downloading from Grok Video Studio")
            setMimeType(mimeType)
            addRequestHeader("User-Agent", userAgent ?: "GrokVideoStudioAndroid/1.0")
            val cookie = CookieManager.getInstance().getCookie(url)
            if (!cookie.isNullOrBlank()) addRequestHeader("Cookie", cookie)
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName)
            setAllowedOverMetered(true)
            setAllowedOverRoaming(true)
        }
        val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        dm.enqueue(request)
        Toast.makeText(this, "Download started: $fileName", Toast.LENGTH_SHORT).show()
    }

    @Composable
    private fun GrokVideoStudioWebApp(
        appUrl: String,
        launchFilePicker: (ValueCallback<Array<Uri>>, Boolean) -> Unit,
        onExternalUrlRequested: (Uri) -> Unit,
        onDownloadRequested: (String, String?, String?, String?) -> Unit,
    ) {
        var isLoading by remember { mutableStateOf(true) }
        var loadProgress by remember { mutableFloatStateOf(0f) }
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
                        settings.databaseEnabled = true
                        settings.loadsImagesAutomatically = true
                        settings.mediaPlaybackRequiresUserGesture = false
                        settings.cacheMode = WebSettings.LOAD_DEFAULT
                        settings.allowFileAccess = true
                        settings.allowContentAccess = true
                        settings.javaScriptCanOpenWindowsAutomatically = true
                        settings.builtInZoomControls = false
                        settings.displayZoomControls = false
                        settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                        settings.userAgentString = settings.userAgentString + " GrokVideoStudioAndroid/1.1"

                        webViewClient = object : WebViewClient() {
                            override fun shouldOverrideUrlLoading(
                                view: WebView?,
                                request: WebResourceRequest?
                            ): Boolean {
                                val uri = request?.url ?: return false
                                val scheme = uri.scheme.orEmpty()
                                if (scheme == "http" || scheme == "https") return false
                                onExternalUrlRequested(uri)
                                return true
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
                                launchFilePicker(
                                    filePathCallback,
                                    fileChooserParams?.mode == FileChooserParams.MODE_OPEN_MULTIPLE,
                                )
                                return true
                            }

                            override fun onPermissionRequest(request: PermissionRequest?) {
                                request?.grant(request.resources)
                            }

                            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                                loadProgress = (newProgress / 100f).coerceIn(0f, 1f)
                                isLoading = newProgress < 100
                            }
                        }

                        setDownloadListener(DownloadListener { url, userAgent, contentDisposition, mimeType, _ ->
                            onDownloadRequested(url, userAgent, contentDisposition, mimeType)
                        })

                        loadUrl(appUrl)
                    }
                },
                update = { webView = it }
            )

            if (isLoading || loadProgress < 1f) {
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            }
        }

        DisposableEffect(webView) {
            onDispose {
                webView?.stopLoading()
                webView?.onPause()
                webView?.pauseTimers()
                webView?.destroy()
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
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
