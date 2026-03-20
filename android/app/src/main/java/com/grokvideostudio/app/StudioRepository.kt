package com.grokvideostudio.app

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.OpenableColumns
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.Transformer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Base64
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class StudioRepository(private val context: Context) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(240, TimeUnit.SECONDS)
        .writeTimeout(240, TimeUnit.SECONDS)
        .build()

    suspend fun generatePrompt(settings: StudioSettings): PromptGenerationResult = withContext(Dispatchers.IO) {
        val concept = settings.concept.trim().ifBlank { settings.manualPrompt.trim() }
        if (concept.isBlank()) {
            throw IllegalArgumentException("Add a concept or manual prompt before generating metadata.")
        }

        if (settings.promptSource == PromptSource.MANUAL) {
            val prompt = settings.manualPrompt.trim().ifBlank { DEFAULT_MANUAL_PROMPT }
            return@withContext PromptGenerationResult(
                manualPrompt = prompt,
                metadata = defaultMetadata(concept, prompt),
                rawResponse = prompt,
            )
        }

        val instruction = "$concept please turn this into a detailed ${settings.videoDurationSeconds} second prompt for grok imagine"
        val system = "You are an expert prompt and social metadata generator for short-form AI videos. Return strict JSON only."
        val user = buildString {
            append("Generate JSON with keys: manual_prompt, title, medium_title, tiktok_subheading, description, x_post, hashtags, category. ")
            append("manual_prompt should be detailed and cinematic for a short AI video. ")
            append("title should be short and catchy. description should be 1-3 sentences. ")
            append("medium_title should be a medium-length title fit for social display. ")
            append("tiktok_subheading should be a slogan/subheading near 120 characters. ")
            append("x_post should be a standalone X-ready post no longer than 275 characters total including spaces and hashtags. ")
            append("hashtags should be an array of 5-12 hashtag strings without # prefixes. ")
            append("category should be the best YouTube category id as a string (default 22 if unsure). ")
            append("Concept instruction: ")
            append(instruction)
        }

        val raw = when (settings.promptSource) {
            PromptSource.GROK -> callGrokChat(settings, system, user)
            PromptSource.OPENAI -> callOpenAiResponses(settings, system, user)
            PromptSource.MANUAL -> settings.manualPrompt
        }
        val metadata = parseSocialMetadata(raw)
        PromptGenerationResult(
            manualPrompt = metadata.manualPrompt,
            metadata = metadata.toSocialMetadata(),
            rawResponse = raw,
        )
    }

    suspend fun generateVideo(
        settings: StudioSettings,
        onStatus: (String) -> Unit,
    ): GeneratedVideo = withContext(Dispatchers.IO) {
        val prompt = prependStylingPrompt(settings.stylingPrompt, settings.manualPrompt)
            .ifBlank { throw IllegalArgumentException("Main prompt is empty.") }

        onStatus("Preparing ${settings.videoProvider.label} request...")
        val (videoUrl, sourceUrl) = when (settings.videoProvider) {
            VideoProvider.GROK -> generateWithGrok(settings, prompt, onStatus)
            VideoProvider.OPENAI -> generateWithOpenAi(settings, prompt, onStatus)
            VideoProvider.SEEDANCE -> generateWithSeedance(settings, prompt, onStatus)
        }

        onStatus("Downloading video to device...")
        val targetFile = createOutputFile(settings.videoProvider.name.lowercase(Locale.US), settings.videoResolution)
        if (videoUrl == OPENAI_CONTENT_SENTINEL) {
            downloadOpenAiContent(settings, requireNotNull(sourceUrl), targetFile)
        } else {
            downloadToFile(videoUrl, targetFile)
        }

        val metadata = settings.socialMetadata.takeIf {
            it.title.isNotBlank() || it.description.isNotBlank() || it.hashtags.isNotEmpty()
        } ?: defaultMetadata(settings.concept, prompt)

        GeneratedVideo(
            id = UUID.randomUUID().toString(),
            title = metadata.title.ifBlank { fallbackTitle(settings.concept, prompt) },
            prompt = prompt,
            provider = settings.videoProvider.label,
            filePath = targetFile.absolutePath,
            sourceUrl = sourceUrl.orEmpty(),
            resolution = resolveOutputSize(settings.videoResolution, settings.videoAspectRatio),
            description = metadata.description,
            hashtags = metadata.hashtags,
            category = metadata.category.ifBlank { DEFAULT_YOUTUBE_CATEGORY },
        )
    }

    suspend fun importVideo(uri: Uri): GeneratedVideo = withContext(Dispatchers.IO) {
        val displayName = queryDisplayName(uri).ifBlank {
            "imported_${timestampStamp()}.mp4"
        }
        val extension = displayName.substringAfterLast('.', "mp4")
        val safeName = slugify(displayName.substringBeforeLast('.', displayName))
        val target = File(mediaDirectory(), "${safeName}_${timestampStamp()}.$extension")
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(target).use { output ->
                input.copyTo(output)
            }
        } ?: throw IOException("Could not read the selected video.")

        GeneratedVideo(
            id = UUID.randomUUID().toString(),
            title = displayName.substringBeforeLast('.', displayName),
            prompt = "Imported from device",
            provider = "Imported",
            filePath = target.absolutePath,
            resolution = "Local file",
        )
    }

    fun mediaDirectory(): File {
        val external = context.getExternalFilesDir(Environment.DIRECTORY_MOVIES)
        val base = external ?: File(context.filesDir, "movies")
        return File(base, "GrokVideoStudio").apply { mkdirs() }
    }

    @UnstableApi
    @Suppress("DEPRECATION")
    suspend fun stitchVideos(
        videos: List<GeneratedVideo>,
        onStatus: (String) -> Unit,
    ): GeneratedVideo {
        require(videos.size >= 2) { "Select at least two clips to stitch." }
        val inputFiles = videos.map { video ->
            File(video.filePath).also { file ->
                if (!file.exists()) {
                    throw IOException("Missing video file for ${video.title}.")
                }
            }
        }
        val outputFile = createOutputFile("stitched", "sequence")
        val stitchedTitle = "Stitched ${timestampStamp()}"
        val stitchedDescription = videos.joinToString(" + ") { it.title }
        val stitchedHashtags = videos.flatMap { it.hashtags }.distinct().take(8)
        val stitchedCategory = videos.firstOrNull()?.category.orEmpty().ifBlank { DEFAULT_YOUTUBE_CATEGORY }

        onStatus("Preparing ${videos.size} clips for export...")
        return withContext(Dispatchers.Main) {
            suspendCancellableCoroutine { continuation ->
                val editedItems = inputFiles.map { file ->
                    EditedMediaItem.Builder(MediaItem.fromUri(Uri.fromFile(file))).build()
                }
                // Media3 1.6.1 still exposes sequence creation through this list constructor.
                val sequence = EditedMediaItemSequence(editedItems)
                val composition: Composition = Composition.Builder(sequence).build()
                val transformer = Transformer.Builder(context)
                    .setVideoMimeType(MimeTypes.VIDEO_H264)
                    .setAudioMimeType(MimeTypes.AUDIO_AAC)
                    .addListener(
                        object : Transformer.Listener {
                            override fun onCompleted(composition: Composition, result: ExportResult) {
                                if (continuation.isActive) {
                                    continuation.resume(
                                        GeneratedVideo(
                                            id = UUID.randomUUID().toString(),
                                            title = stitchedTitle,
                                            prompt = "Stitched from ${videos.joinToString(", ") { it.title }}",
                                            provider = "Stitched",
                                            filePath = outputFile.absolutePath,
                                            sourceUrl = outputFile.absolutePath,
                                            resolution = videos.firstOrNull()?.resolution.orEmpty().ifBlank { "Stitched export" },
                                            description = stitchedDescription,
                                            hashtags = stitchedHashtags,
                                            category = stitchedCategory,
                                        ),
                                    )
                                }
                            }

                            override fun onError(
                                composition: Composition,
                                result: ExportResult,
                                exception: ExportException,
                            ) {
                                if (continuation.isActive) {
                                    continuation.resumeWithException(exception)
                                }
                            }
                        },
                    )
                    .build()

                continuation.invokeOnCancellation {
                    transformer.cancel()
                    outputFile.delete()
                }
                onStatus("Exporting stitched video...")
                transformer.start(composition, outputFile.absolutePath)
            }
        }
    }

    private fun generateWithGrok(
        settings: StudioSettings,
        prompt: String,
        onStatus: (String) -> Unit,
    ): Pair<String, String?> {
        val apiKey = settings.grokApiKey.trim()
        if (apiKey.isBlank()) {
            throw IllegalArgumentException("Add your Grok API key in Settings first.")
        }

        val size = resolveOutputSize(settings.videoResolution, settings.videoAspectRatio)
        val response = postJson(
            url = "$XAI_API_BASE/imagine/video/generations",
            headers = mapOf(
                "Authorization" to "Bearer $apiKey",
                "Content-Type" to "application/json",
            ),
            payload = JSONObject()
                .put("model", settings.grokVideoModel.trim().ifBlank { "grok-video-latest" })
                .put("prompt", prompt)
                .put("duration_seconds", settings.videoDurationSeconds)
                .put("resolution", size)
                .put("fps", 24),
        )
        val jobId = response.optString("id").trim()
        if (jobId.isBlank()) {
            throw IOException("Grok Imagine did not return a generation id.")
        }

        repeat(POLL_ATTEMPTS) { attempt ->
            onStatus("Polling Grok Imagine job ${attempt + 1}/$POLL_ATTEMPTS...")
            val payload = getJson(
                url = "$XAI_API_BASE/imagine/video/generations/$jobId",
                headers = mapOf("Authorization" to "Bearer $apiKey"),
            )
            val status = payload.optString("status").lowercase(Locale.US)
            when (status) {
                "succeeded", "completed", "ready" -> {
                    val output = payload.optJSONObject("output")
                    val videoUrl = output?.optString("video_url").orEmpty().ifBlank {
                        payload.optString("video_url")
                    }
                    if (videoUrl.isBlank()) {
                        throw IOException("Grok Imagine completed but returned no downloadable video URL.")
                    }
                    return videoUrl to videoUrl
                }

                "failed", "error" -> throw IOException(payload.optString("error").ifBlank { "Grok generation failed." })
            }
            Thread.sleep(POLL_INTERVAL_MS)
        }
        throw IOException("Timed out waiting for Grok Imagine to finish.")
    }

    private suspend fun generateWithOpenAi(
        settings: StudioSettings,
        prompt: String,
        onStatus: (String) -> Unit,
    ): Pair<String, String?> {
        val headers = openAiHeaders(settings)
        val size = resolveOutputSize(settings.videoResolution, settings.videoAspectRatio)
        val payload = JSONObject()
            .put("model", settings.openAiSoraModel.trim().ifBlank { "sora-2" })
            .put("prompt", prompt)
            .put("size", size)
            .put("seconds", settings.videoDurationSeconds.toString())

        var lastError = ""
        val endpoints = listOf(
            "$OPENAI_API_BASE/videos/generations",
            "$OPENAI_API_BASE/videos",
        )

        for (endpoint in endpoints) {
            val response = tryPostJson(endpoint, headers, payload)
            if (response == null) {
                continue
            }
            val jobId = response.optString("id").trim().ifBlank { response.optString("job_id").trim() }
            val directUrl = extractVideoUrl(response)
            if (directUrl.isNotBlank()) {
                return directUrl to directUrl
            }
            if (jobId.isBlank()) {
                lastError = "OpenAI did not return a job id or a direct video URL."
                continue
            }

            repeat(POLL_ATTEMPTS) { attempt ->
                onStatus("Polling OpenAI video job ${attempt + 1}/$POLL_ATTEMPTS...")
                val pollPayload = pollVideoJob(
                    headers = headers,
                    endpoints = listOf(
                        "$OPENAI_API_BASE/videos/generations/$jobId",
                        "$OPENAI_API_BASE/videos/$jobId",
                    ),
                )
                val videoUrl = extractVideoUrl(pollPayload)
                if (videoUrl.isNotBlank()) {
                    return videoUrl to videoUrl
                }
                val status = pollPayload.optString("status").lowercase(Locale.US)
                if (status in setOf("succeeded", "completed", "ready")) {
                    return OPENAI_CONTENT_SENTINEL to jobId
                }
                delay(POLL_INTERVAL_MS)
            }

            lastError = "Timed out waiting for OpenAI video generation."
        }

        throw IOException("OpenAI Sora request failed. ${lastError.ifBlank { "No compatible endpoint was available." }}")
    }

    private suspend fun generateWithSeedance(
        settings: StudioSettings,
        prompt: String,
        onStatus: (String) -> Unit,
    ): Pair<String, String?> {
        val credential = settings.seedanceApiKey.trim().ifBlank { settings.seedanceOauthToken.trim() }
        if (credential.isBlank()) {
            throw IllegalArgumentException("Add your Seedance API key or OAuth token in Settings first.")
        }
        val headers = mapOf(
            "Authorization" to "Bearer $credential",
            "Content-Type" to "application/json",
        )
        val payload = JSONObject()
            .put("model", settings.seedanceModel.trim().ifBlank { "seedance-2.0" })
            .put("prompt", prompt)
            .put("duration_seconds", settings.videoDurationSeconds)
            .put("resolution", resolveOutputSize(settings.videoResolution, settings.videoAspectRatio))
            .put("aspect_ratio", settings.videoAspectRatio)
            .put("fps", 24)
            .put("motion_strength", 0.6)
            .put("guidance_scale", 7.5)
            .put("watermark", false)

        var lastError = ""
        val endpoints = listOf(
            "$SEEDANCE_API_BASE/videos/generations",
            "$SEEDANCE_API_BASE/videos",
            "$SEEDANCE_API_BASE/video/generations",
        )

        for (endpoint in endpoints) {
            val response = tryPostJson(endpoint, headers, payload)
            if (response == null) {
                continue
            }
            val jobId = response.optString("id").trim().ifBlank { response.optString("job_id").trim() }
            val directUrl = extractVideoUrl(response)
            if (directUrl.isNotBlank()) {
                return directUrl to directUrl
            }
            if (jobId.isBlank()) {
                lastError = "Seedance did not return a job id or a direct video URL."
                continue
            }

            repeat(POLL_ATTEMPTS) { attempt ->
                onStatus("Polling Seedance job ${attempt + 1}/$POLL_ATTEMPTS...")
                val pollPayload = pollVideoJob(
                    headers = headers,
                    endpoints = listOf(
                        "$SEEDANCE_API_BASE/videos/generations/$jobId",
                        "$SEEDANCE_API_BASE/videos/$jobId",
                        "$SEEDANCE_API_BASE/video/generations/$jobId",
                    ),
                )
                val videoUrl = extractVideoUrl(pollPayload)
                if (videoUrl.isNotBlank()) {
                    return videoUrl to videoUrl
                }
                delay(POLL_INTERVAL_MS)
            }
            lastError = "Timed out waiting for Seedance video generation."
        }

        throw IOException("Seedance request failed. ${lastError.ifBlank { "No compatible endpoint was available." }}")
    }

    private suspend fun pollVideoJob(
        headers: Map<String, String>,
        endpoints: List<String>,
    ): JSONObject {
        for (endpoint in endpoints) {
            val payload = tryGetJson(endpoint, headers) ?: continue
            val status = payload.optString("status").lowercase(Locale.US)
            if (status in setOf("failed", "error", "cancelled", "canceled")) {
                throw IOException(payload.optString("error").ifBlank { "Video generation failed." })
            }
            return payload
        }
        throw IOException("No compatible polling endpoint responded.")
    }

    private fun callGrokChat(settings: StudioSettings, system: String, user: String): String {
        val apiKey = settings.grokApiKey.trim()
        if (apiKey.isBlank()) {
            throw IllegalArgumentException("Add your Grok API key in Settings first.")
        }
        val payload = JSONObject()
            .put("model", settings.grokChatModel.trim().ifBlank { "grok-3-mini" })
            .put("temperature", 0.9)
            .put(
                "messages",
                JSONArray()
                    .put(JSONObject().put("role", "system").put("content", system))
                    .put(JSONObject().put("role", "user").put("content", user)),
            )
        val response = postJson(
            url = "$XAI_API_BASE/chat/completions",
            headers = mapOf(
                "Authorization" to "Bearer $apiKey",
                "Content-Type" to "application/json",
            ),
            payload = payload,
        )
        return response.optJSONArray("choices")
            ?.optJSONObject(0)
            ?.optJSONObject("message")
            ?.optString("content")
            .orEmpty()
            .trim()
            .ifBlank { throw IOException("Grok did not return prompt text.") }
    }

    private fun callOpenAiResponses(settings: StudioSettings, system: String, user: String): String {
        val response = postJson(
            url = "$OPENAI_API_BASE/responses",
            headers = openAiHeaders(settings),
            payload = JSONObject()
                .put("model", settings.openAiChatModel.trim().ifBlank { "gpt-5.1-codex" })
                .put("instructions", system)
                .put("store", false)
                .put("max_output_tokens", 1200)
                .put(
                    "input",
                    JSONArray().put(
                        JSONObject()
                            .put("role", "user")
                            .put("content", user),
                    ),
                ),
        )
        val directText = response.optString("output_text").trim()
        if (directText.isNotBlank()) {
            return directText
        }

        val output = response.optJSONArray("output") ?: JSONArray()
        for (i in 0 until output.length()) {
            val item = output.optJSONObject(i) ?: continue
            val content = item.optJSONArray("content") ?: continue
            for (j in 0 until content.length()) {
                val entry = content.optJSONObject(j) ?: continue
                val text = entry.optString("text").trim()
                if (text.isNotBlank()) {
                    return text
                }
            }
        }
        throw IOException("OpenAI did not return prompt text.")
    }

    private fun parseSocialMetadata(raw: String): ParsedSocialMetadata {
        val json = JSONObject(extractFirstJsonObject(raw))
        val manualPrompt = json.optString("manual_prompt").trim()
        if (manualPrompt.isBlank()) {
            throw IOException("AI response did not include manual_prompt.")
        }
        val hashtags = when (val rawHashtags = json.opt("hashtags")) {
            is JSONArray -> buildList {
                for (index in 0 until rawHashtags.length()) {
                    val value = rawHashtags.optString(index).trim().removePrefix("#")
                    if (value.isNotBlank()) add(value)
                }
            }

            is String -> rawHashtags
                .split(',', ' ', '\n', '\t')
                .map { it.trim().removePrefix("#") }
                .filter { it.isNotBlank() }

            else -> emptyList()
        }
        return ParsedSocialMetadata(
            manualPrompt = manualPrompt,
            title = json.optString("title").trim().ifBlank { fallbackTitle("", manualPrompt) },
            mediumTitle = json.optString("medium_title").trim(),
            tiktokSubheading = json.optString("tiktok_subheading").trim(),
            description = json.optString("description").trim(),
            xPost = json.optString("x_post").trim(),
            hashtags = hashtags,
            category = json.optString("category").trim().ifBlank { DEFAULT_YOUTUBE_CATEGORY },
        )
    }

    private fun ParsedSocialMetadata.toSocialMetadata(): SocialMetadata = SocialMetadata(
        title = title,
        mediumTitle = mediumTitle.ifBlank { title },
        tiktokSubheading = tiktokSubheading.ifBlank { title },
        description = description,
        xPost = xPost.ifBlank {
            buildString {
                append(title)
                if (hashtags.isNotEmpty()) {
                    append('\n')
                    append(hashtags.joinToString(" ") { "#$it" })
                }
            }
        },
        hashtags = hashtags,
        category = category,
    )

    private fun defaultMetadata(concept: String, prompt: String): SocialMetadata {
        val baseTitle = fallbackTitle(concept, prompt)
        val tags = concept
            .split(' ', ',', '.', '\n')
            .map { it.trim().lowercase(Locale.US).removePrefix("#") }
            .filter { it.length in 3..18 }
            .distinct()
            .take(5)
        return SocialMetadata(
            title = baseTitle,
            mediumTitle = baseTitle,
            tiktokSubheading = baseTitle,
            description = concept.trim().ifBlank { prompt.take(180) },
            xPost = listOf(baseTitle, tags.joinToString(" ") { "#$it" })
                .filter { it.isNotBlank() }
                .joinToString("\n"),
            hashtags = tags,
            category = DEFAULT_YOUTUBE_CATEGORY,
        )
    }

    private fun fallbackTitle(concept: String, prompt: String): String {
        val base = concept.ifBlank { prompt }.replace(Regex("\\s+"), " ").trim()
        if (base.isBlank()) return "AI Video"
        return base
            .split(' ')
            .take(6)
            .joinToString(" ")
            .replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.US) else it.toString() }
    }

    private fun prependStylingPrompt(styling: String, prompt: String): String {
        val left = styling.trim()
        val right = prompt.trim()
        return when {
            left.isBlank() -> right
            right.isBlank() -> left
            else -> "$left $right"
        }.trim()
    }

    private fun openAiHeaders(settings: StudioSettings): Map<String, String> {
        val credential = settings.openAiApiKey.trim().ifBlank { settings.openAiAccessToken.trim() }
        if (credential.isBlank()) {
            throw IllegalArgumentException("Add your OpenAI API key or access token in Settings first.")
        }
        val headers = mutableMapOf(
            "Authorization" to "Bearer $credential",
            "Content-Type" to "application/json",
        )
        val claims = decodeJwtClaims(credential)
        val auth = claims?.optJSONObject("https://api.openai.com/auth")
        val orgId = firstString(auth, "organization_id", "org_id", "organization")
            .ifBlank { firstString(claims, "organization_id", "org_id", "organization") }
        val projectId = firstString(auth, "project_id", "project")
            .ifBlank { firstString(claims, "project_id", "project") }
        if (orgId.isNotBlank()) {
            headers["OpenAI-Organization"] = orgId
        }
        if (projectId.isNotBlank()) {
            headers["OpenAI-Project"] = projectId
        }
        return headers
    }

    private fun decodeJwtClaims(credential: String): JSONObject? {
        val parts = credential.split('.')
        if (parts.size < 2) {
            return null
        }
        return runCatching {
            val decoded = String(Base64.getUrlDecoder().decode(parts[1]))
            JSONObject(decoded)
        }.getOrNull()
    }

    private fun firstString(source: JSONObject?, vararg keys: String): String {
        if (source == null) {
            return ""
        }
        for (key in keys) {
            val value = source.optString(key).trim()
            if (value.isNotBlank()) {
                return value
            }
        }
        return ""
    }

    private fun resolveOutputSize(resolution: String, aspectRatio: String): String {
        val parts = resolution.split('x')
        val width = parts.getOrNull(0)?.toIntOrNull() ?: 1280
        val height = parts.getOrNull(1)?.toIntOrNull() ?: 720
        return when (aspectRatio) {
            "9:16" -> "${height}x${width}"
            "1:1" -> {
                val edge = minOf(width, height)
                "${edge}x${edge}"
            }

            else -> "${width}x${height}"
        }
    }

    private fun createOutputFile(prefix: String, resolution: String): File {
        val safePrefix = slugify(prefix)
        val safeResolution = resolution.replace(':', '-')
        return File(mediaDirectory(), "${safePrefix}_${safeResolution}_${timestampStamp()}.mp4")
    }

    private fun timestampStamp(): String =
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

    private fun slugify(value: String): String =
        value.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "_").trim('_').ifBlank { "video" }

    private fun queryDisplayName(uri: Uri): String {
        return context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0).orEmpty() else ""
            }
            .orEmpty()
    }

    private fun downloadOpenAiContent(settings: StudioSettings, jobId: String, targetFile: File) {
        val headers = openAiHeaders(settings)
        val endpoints = listOf(
            "$OPENAI_API_BASE/videos/$jobId/content",
            "$OPENAI_API_BASE/videos/generations/$jobId/content",
        )
        for (endpoint in endpoints) {
            if (tryDownloadAuthorized(endpoint, headers, targetFile)) {
                return
            }
        }
        throw IOException("OpenAI Sora completed but the content endpoint did not return a video file.")
    }

    private fun downloadToFile(url: String, targetFile: File) {
        val request = Request.Builder().url(url).get().build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Download failed: ${response.code} ${response.body?.string().orEmpty().take(300)}")
            }
            val body = response.body ?: throw IOException("Download returned an empty body.")
            targetFile.outputStream().use { output ->
                body.byteStream().use { input -> input.copyTo(output) }
            }
        }
    }

    private fun tryDownloadAuthorized(url: String, headers: Map<String, String>, targetFile: File): Boolean {
        val requestBuilder = Request.Builder().url(url).get()
        headers.forEach { (key, value) -> requestBuilder.header(key, value) }
        client.newCall(requestBuilder.build()).execute().use { response ->
            if (response.code in setOf(404, 405, 501)) {
                return false
            }
            if (!response.isSuccessful) {
                throw IOException("Authorized download failed: ${response.code} ${response.body?.string().orEmpty().take(300)}")
            }
            val body = response.body ?: throw IOException("Download returned an empty body.")
            targetFile.outputStream().use { output ->
                body.byteStream().use { input -> input.copyTo(output) }
            }
            return true
        }
    }

    private fun extractVideoUrl(payload: JSONObject): String {
        val topLevel = listOf("video_url", "url")
            .asSequence()
            .map { payload.optString(it).trim() }
            .firstOrNull { it.isNotBlank() }
        if (!topLevel.isNullOrBlank()) {
            return topLevel
        }

        val output = payload.optJSONObject("output")
        val nestedOutput = output?.optString("video_url").orEmpty().trim()
        if (nestedOutput.isNotBlank()) {
            return nestedOutput
        }

        val data = payload.optJSONArray("data") ?: return ""
        for (index in 0 until data.length()) {
            val item = data.optJSONObject(index) ?: continue
            val direct = item.optString("video_url").trim().ifBlank { item.optString("url").trim() }
            if (direct.isNotBlank()) {
                return direct
            }
            val itemOutput = item.optJSONObject("output")?.optString("video_url").orEmpty().trim()
            if (itemOutput.isNotBlank()) {
                return itemOutput
            }
        }
        return ""
    }

    private fun extractFirstJsonObject(raw: String): String {
        var depth = 0
        var start = -1
        var inString = false
        var escaped = false
        raw.forEachIndexed { index, ch ->
            when {
                escaped -> escaped = false
                ch == '\\' && inString -> escaped = true
                ch == '"' -> inString = !inString
                inString -> Unit
                ch == '{' -> {
                    if (depth == 0) {
                        start = index
                    }
                    depth += 1
                }

                ch == '}' -> {
                    if (depth > 0) {
                        depth -= 1
                        if (depth == 0 && start >= 0) {
                            return raw.substring(start, index + 1)
                        }
                    }
                }
            }
        }
        throw IOException("Could not find a JSON object in the AI response.")
    }

    private fun postJson(url: String, headers: Map<String, String>, payload: JSONObject): JSONObject {
        val requestBuilder = Request.Builder()
            .url(url)
            .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
        headers.forEach { (key, value) -> requestBuilder.header(key, value) }
        client.newCall(requestBuilder.build()).execute().use { response ->
            val bodyText = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IOException("${response.code} ${bodyText.take(500)}")
            }
            return if (bodyText.isBlank()) JSONObject() else JSONObject(bodyText)
        }
    }

    private fun tryPostJson(url: String, headers: Map<String, String>, payload: JSONObject): JSONObject? {
        val requestBuilder = Request.Builder()
            .url(url)
            .post(payload.toString().toRequestBody(JSON_MEDIA_TYPE))
        headers.forEach { (key, value) -> requestBuilder.header(key, value) }
        client.newCall(requestBuilder.build()).execute().use { response ->
            val bodyText = response.body?.string().orEmpty()
            if (response.code in setOf(404, 405, 501)) {
                return null
            }
            if (!response.isSuccessful) {
                throw IOException("${response.code} ${bodyText.take(500)}")
            }
            return if (bodyText.isBlank()) JSONObject() else JSONObject(bodyText)
        }
    }

    private fun getJson(url: String, headers: Map<String, String>): JSONObject {
        val requestBuilder = Request.Builder().url(url).get()
        headers.forEach { (key, value) -> requestBuilder.header(key, value) }
        client.newCall(requestBuilder.build()).execute().use { response ->
            val bodyText = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw IOException("${response.code} ${bodyText.take(500)}")
            }
            return if (bodyText.isBlank()) JSONObject() else JSONObject(bodyText)
        }
    }

    private fun tryGetJson(url: String, headers: Map<String, String>): JSONObject? {
        val requestBuilder = Request.Builder().url(url).get()
        headers.forEach { (key, value) -> requestBuilder.header(key, value) }
        client.newCall(requestBuilder.build()).execute().use { response ->
            val bodyText = response.body?.string().orEmpty()
            if (response.code in setOf(404, 405, 501)) {
                return null
            }
            if (!response.isSuccessful) {
                throw IOException("${response.code} ${bodyText.take(500)}")
            }
            return if (bodyText.isBlank()) JSONObject() else JSONObject(bodyText)
        }
    }

    private data class ParsedSocialMetadata(
        val manualPrompt: String,
        val title: String,
        val mediumTitle: String,
        val tiktokSubheading: String,
        val description: String,
        val xPost: String,
        val hashtags: List<String>,
        val category: String,
    )

    private companion object {
        const val XAI_API_BASE = "https://api.x.ai/v1"
        const val OPENAI_API_BASE = "https://api.openai.com/v1"
        const val SEEDANCE_API_BASE = "https://api.seedance.ai/v2"
        const val OPENAI_CONTENT_SENTINEL = "__openai_content__"
        const val POLL_ATTEMPTS = 180
        const val POLL_INTERVAL_MS = 5000L
        val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}
