package com.grokvideostudio.app

import androidx.annotation.Keep

const val DEFAULT_MANUAL_PROMPT =
    "abstract surreal artistic photorealistic strange random dream like scifi fast moving camera, fast moving fractals morphing and intersecting, highly detailed"
const val DEFAULT_YOUTUBE_CATEGORY = "22"

@Keep
enum class PromptSource(val label: String) {
    MANUAL("Manual"),
    GROK("Grok API"),
    OPENAI("OpenAI API"),
}

@Keep
enum class VideoProvider(val label: String) {
    GROK("Grok Imagine"),
    OPENAI("OpenAI Sora"),
    SEEDANCE("Seedance 2.0"),
}

@Keep
data class SocialMetadata(
    val title: String = "",
    val mediumTitle: String = "",
    val tiktokSubheading: String = "",
    val description: String = "",
    val xPost: String = "",
    val hashtags: List<String> = emptyList(),
    val category: String = DEFAULT_YOUTUBE_CATEGORY,
)

@Keep
data class StudioSettings(
    val grokApiKey: String = "",
    val grokChatModel: String = "grok-3-mini",
    val grokVideoModel: String = "grok-video-latest",
    val openAiApiKey: String = "",
    val openAiAccessToken: String = "",
    val openAiChatModel: String = "gpt-5.1-codex",
    val openAiSoraModel: String = "sora-2",
    val seedanceApiKey: String = "",
    val seedanceOauthToken: String = "",
    val seedanceModel: String = "seedance-2.0",
    val promptSource: PromptSource = PromptSource.MANUAL,
    val videoProvider: VideoProvider = VideoProvider.OPENAI,
    val concept: String = "",
    val stylingPrompt: String = "",
    val manualPrompt: String = DEFAULT_MANUAL_PROMPT,
    val videoResolution: String = "1280x720",
    val videoAspectRatio: String = "16:9",
    val videoDurationSeconds: Int = 8,
    val socialMetadata: SocialMetadata = SocialMetadata(),
)

@Keep
data class GeneratedVideo(
    val id: String,
    val title: String,
    val prompt: String,
    val provider: String,
    val filePath: String,
    val sourceUrl: String = "",
    val resolution: String = "",
    val createdAtEpochMs: Long = System.currentTimeMillis(),
    val description: String = "",
    val hashtags: List<String> = emptyList(),
    val category: String = DEFAULT_YOUTUBE_CATEGORY,
)

data class PromptGenerationResult(
    val manualPrompt: String,
    val metadata: SocialMetadata,
    val rawResponse: String,
)

data class StudioUiState(
    val settings: StudioSettings = StudioSettings(),
    val videos: List<GeneratedVideo> = emptyList(),
    val selectedVideoId: String? = null,
    val stitchSelectionIds: Set<String> = emptySet(),
    val publishDraft: SocialMetadata = SocialMetadata(),
    val isBusy: Boolean = false,
    val statusMessage: String = "Ready to create videos on Android.",
    val errorMessage: String? = null,
    val activityLog: List<String> = listOf("Native Android workspace ready."),
)

fun SocialMetadata.hashtagText(): String = hashtags.joinToString(" ") { tag ->
    if (tag.startsWith("#")) tag else "#$tag"
}

fun GeneratedVideo.toSocialMetadata(): SocialMetadata = SocialMetadata(
    title = title,
    mediumTitle = title,
    tiktokSubheading = title,
    description = description,
    xPost = buildString {
        append(title)
        if (description.isNotBlank()) {
            append('\n')
            append(description)
        }
        if (hashtags.isNotEmpty()) {
            append('\n')
            append(hashtags.joinToString(" ") { if (it.startsWith("#")) it else "#$it" })
        }
    }.trim(),
    hashtags = hashtags,
    category = category.ifBlank { DEFAULT_YOUTUBE_CATEGORY },
)
