package com.grokvideostudio.app

import android.annotation.SuppressLint
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import com.grokvideostudio.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    /**
     * Host the public Grok Video Studio page inside a lightweight Android shell.
     * This gives users a fast installable mobile companion while keeping rollout simple.
     */
    private val launchUrl = "https://mysticalg.github.io/Grok-Video-Studio/"

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.retryButton.setOnClickListener {
            loadHomePage(force = true)
        }

        with(binding.webView.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true
            cacheMode = WebSettings.LOAD_DEFAULT
            loadsImagesAutomatically = true
            mediaPlaybackRequiresUserGesture = true
        }

        binding.webView.webChromeClient = WebChromeClient()
        binding.webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                showLoading(false)
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                // Only show a full-screen retry panel when the primary frame fails.
                if (request?.isForMainFrame == true) {
                    showOfflineMessage()
                }
            }
        }

        loadHomePage(force = false)
    }

    private fun loadHomePage(force: Boolean) {
        if (!isNetworkAvailable()) {
            showOfflineMessage()
            return
        }

        showLoading(true)
        if (force) {
            binding.webView.clearCache(true)
        }
        binding.webView.loadUrl(launchUrl)
    }

    private fun showLoading(loading: Boolean) {
        binding.loadingPanel.visibility = if (loading) android.view.View.VISIBLE else android.view.View.GONE
        binding.retryButton.visibility = android.view.View.GONE
        binding.statusText.setText(R.string.loading)
    }

    private fun showOfflineMessage() {
        binding.loadingPanel.visibility = android.view.View.VISIBLE
        binding.retryButton.visibility = android.view.View.VISIBLE
        binding.statusText.setText(R.string.offline_help)
    }

    private fun isNetworkAvailable(): Boolean {
        val connectivityManager = getSystemService(ConnectivityManager::class.java)
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    override fun onBackPressed() {
        if (binding.webView.canGoBack()) {
            binding.webView.goBack()
            return
        }
        super.onBackPressed()
    }
}
