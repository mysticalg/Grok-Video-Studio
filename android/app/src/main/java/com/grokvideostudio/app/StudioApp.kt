package com.grokvideostudio.app

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import android.net.Uri
import android.widget.MediaController
import android.widget.VideoView
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private enum class AppTab(val label: String) {
    CREATE("Create"),
    LIBRARY("Library"),
    PUBLISH("Publish"),
    SETTINGS("Settings"),
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
fun GrokVideoStudioApp(viewModel: StudioViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val tabs = listOf(
        Triple(AppTab.CREATE, "Create", Icons.Outlined.AutoAwesome),
        Triple(AppTab.LIBRARY, "Library", Icons.Outlined.VideoLibrary),
        Triple(AppTab.PUBLISH, "Publish", Icons.Outlined.Share),
        Triple(AppTab.SETTINGS, "Settings", Icons.Outlined.Settings),
    )
    var currentTab by rememberSaveable { mutableStateOf(AppTab.CREATE) }
    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            viewModel.importVideo(uri)
            currentTab = AppTab.LIBRARY
        }
    }

    StudioTheme {
        Scaffold(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding(),
            topBar = {
                CenterAlignedTopAppBar(
                    title = {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text("Grok Video Studio", fontWeight = FontWeight.SemiBold)
                            Text(
                                "Native Android build",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                )
            },
            bottomBar = {
                NavigationBar(modifier = Modifier.navigationBarsPadding()) {
                    tabs.forEach { (tab, label, icon) ->
                        NavigationBarItem(
                            selected = currentTab == tab,
                            onClick = { currentTab = tab },
                            icon = { Icon(icon, contentDescription = label) },
                            label = { Text(label) },
                        )
                    }
                }
            },
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
            ) {
                if (uiState.isBusy) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                uiState.errorMessage?.let { message ->
                    InfoCard(
                        title = "Last action failed",
                        body = message,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                        tone = CardTone.Error,
                        action = {
                            TextButton(onClick = viewModel::clearError) {
                                Text("Dismiss")
                            }
                        },
                    )
                }
                when (currentTab) {
                    AppTab.CREATE -> CreateScreen(
                        uiState = uiState,
                        onSettingsChange = viewModel::updateSettings,
                        onGeneratePrompt = viewModel::generatePrompt,
                        onGenerateVideo = {
                            viewModel.generateVideo()
                            currentTab = AppTab.LIBRARY
                        },
                    )

                    AppTab.LIBRARY -> LibraryScreen(
                        uiState = uiState,
                        onImport = { importLauncher.launch(arrayOf("video/*")) },
                        onSelectVideo = {
                            viewModel.selectVideo(it)
                        },
                        onToggleForStitch = viewModel::toggleVideoForStitch,
                        onStitchSelected = viewModel::stitchSelectedVideos,
                        onDeleteSelected = viewModel::removeSelectedVideo,
                    )

                    AppTab.PUBLISH -> PublishScreen(
                        uiState = uiState,
                        onDraftChange = viewModel::updatePublishDraft,
                    )

                    AppTab.SETTINGS -> SettingsScreen(
                        uiState = uiState,
                        storagePath = viewModel.mediaDirectoryPath(),
                        onSettingsChange = viewModel::updateSettings,
                    )
                }
            }
        }
    }
}

@Composable
private fun CreateScreen(
    uiState: StudioUiState,
    onSettingsChange: ((StudioSettings) -> StudioSettings) -> Unit,
    onGeneratePrompt: () -> Unit,
    onGenerateVideo: () -> Unit,
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            InfoCard(
                title = "Native creation workspace",
                body = "Prompt drafting, API generation, device library, clip stitching, and mobile share hand-off now run inside the Android app instead of a WebView wrapper.",
            )
        }
        item {
            SectionCard(title = "Prompt") {
                OutlinedTextField(
                    value = uiState.settings.concept,
                    onValueChange = { value -> onSettingsChange { it.copy(concept = value) } },
                    label = { Text("Concept") },
                    placeholder = { Text("Describe the scene or idea...") },
                    minLines = 4,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 120.dp),
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.stylingPrompt,
                    onValueChange = { value -> onSettingsChange { it.copy(stylingPrompt = value) } },
                    label = { Text("Style add-on") },
                    placeholder = { Text("Optional style words prepended to the main prompt...") },
                    minLines = 3,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 96.dp),
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.manualPrompt,
                    onValueChange = { value -> onSettingsChange { it.copy(manualPrompt = value) } },
                    label = { Text("Main prompt") },
                    minLines = 6,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 160.dp),
                )
            }
        }
        item {
            SectionCard(title = "Generation setup") {
                EnumChipRow(
                    title = "Prompt source",
                    options = PromptSource.entries.toList(),
                    selected = uiState.settings.promptSource,
                    label = { it.label },
                    onSelected = { choice -> onSettingsChange { it.copy(promptSource = choice) } },
                )
                Spacer(modifier = Modifier.height(12.dp))
                EnumChipRow(
                    title = "Video provider",
                    options = VideoProvider.entries.toList(),
                    selected = uiState.settings.videoProvider,
                    label = { it.label },
                    onSelected = { choice -> onSettingsChange { it.copy(videoProvider = choice) } },
                )
                Spacer(modifier = Modifier.height(12.dp))
                StringChipRow(
                    title = "Resolution",
                    options = listOf("854x480", "1280x720"),
                    selected = uiState.settings.videoResolution,
                    onSelected = { value -> onSettingsChange { it.copy(videoResolution = value) } },
                )
                Spacer(modifier = Modifier.height(12.dp))
                StringChipRow(
                    title = "Aspect ratio",
                    options = listOf("16:9", "9:16", "1:1"),
                    selected = uiState.settings.videoAspectRatio,
                    onSelected = { value -> onSettingsChange { it.copy(videoAspectRatio = value) } },
                )
                Spacer(modifier = Modifier.height(12.dp))
                StringChipRow(
                    title = "Duration",
                    options = listOf("6", "8", "10", "12"),
                    selected = uiState.settings.videoDurationSeconds.toString(),
                    label = { "${it}s" },
                    onSelected = { value ->
                        onSettingsChange { it.copy(videoDurationSeconds = value.toIntOrNull() ?: 8) }
                    },
                )
                Spacer(modifier = Modifier.height(16.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedButton(onClick = onGeneratePrompt, modifier = Modifier.weight(1f)) {
                        Text("Generate prompt")
                    }
                    Button(onClick = onGenerateVideo, modifier = Modifier.weight(1f)) {
                        Text("Generate video")
                    }
                }
            }
        }
        item {
            SectionCard(title = "Metadata preview") {
                Text(
                    uiState.publishDraft.title.ifBlank { "No title yet" },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    uiState.publishDraft.description.ifBlank { "Generate a prompt or edit the publish draft later in the Publish tab." },
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (uiState.publishDraft.hashtags.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        uiState.publishDraft.hashtagText(),
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }
        item {
            InfoCard(
                title = "Desktop extras",
                body = "Browser automation uploaders, AI Flow Trainer, and the advanced desktop stitch pipeline with crossfade, interpolation, upscale, and music mixing still need a later Android-native port or backend companion.",
            )
        }
        item {
            SectionCard(title = "Activity") {
                Text(uiState.statusMessage, fontWeight = FontWeight.Medium)
                Spacer(modifier = Modifier.height(12.dp))
                uiState.activityLog.take(8).forEach { line ->
                    Text(
                        line,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(bottom = 4.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun LibraryScreen(
    uiState: StudioUiState,
    onImport: () -> Unit,
    onSelectVideo: (String) -> Unit,
    onToggleForStitch: (String) -> Unit,
    onStitchSelected: () -> Unit,
    onDeleteSelected: () -> Unit,
) {
    val context = LocalContext.current
    val selected = uiState.videos.firstOrNull { it.id == uiState.selectedVideoId }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            SectionCard(title = "Device library") {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = onImport, modifier = Modifier.weight(1f)) {
                        Text("Import from phone")
                    }
                    OutlinedButton(
                        onClick = onDeleteSelected,
                        enabled = selected != null,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text("Remove selected")
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))
                Button(
                    onClick = onStitchSelected,
                    enabled = uiState.stitchSelectionIds.size >= 2,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        if (uiState.stitchSelectionIds.isEmpty()) {
                            "Select clips to stitch"
                        } else {
                            "Stitch ${uiState.stitchSelectionIds.size} selected clips"
                        },
                    )
                }
            }
        }
        selected?.let { video ->
            item {
                SectionCard(title = video.title) {
                    VideoPreview(video = video)
                    Spacer(modifier = Modifier.height(12.dp))
                    Text("${video.provider} - ${video.resolution}", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (video.description.isNotBlank()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(video.description)
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Button(
                            onClick = { openVideo(context, video) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Open")
                        }
                        OutlinedButton(
                            onClick = { shareVideo(context, video, video.toSocialMetadata(), null) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Share")
                        }
                    }
                }
            }
        }
        if (uiState.videos.isEmpty()) {
            item {
                InfoCard(
                    title = "No videos yet",
                    body = "Generate a clip from Create or import a local MP4 to populate the native Android library.",
                )
            }
        } else {
            items(uiState.videos, key = { it.id }) { video ->
                ElevatedCard(onClick = { onSelectVideo(video.id) }) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        VideoThumbnail(
                            video = video,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(180.dp),
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = video.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "${video.provider} - ${video.resolution}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = video.prompt,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        FilterChip(
                            selected = video.id in uiState.stitchSelectionIds,
                            onClick = { onToggleForStitch(video.id) },
                            label = {
                                Text(
                                    if (video.id in uiState.stitchSelectionIds) {
                                        "Included in stitch"
                                    } else {
                                        "Add to stitch"
                                    },
                                )
                            },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PublishScreen(
    uiState: StudioUiState,
    onDraftChange: ((SocialMetadata) -> SocialMetadata) -> Unit,
) {
    val context = LocalContext.current
    val selected = uiState.videos.firstOrNull { it.id == uiState.selectedVideoId }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            InfoCard(
                title = "Mobile publish draft",
                body = "This tab now prepares real captions in the app. Share-target hand-off is native too, but social API uploads still need a later mobile-specific implementation.",
            )
        }
        if (selected == null) {
            item {
                InfoCard(
                    title = "Select a video first",
                    body = "Choose a generated or imported clip in Library before preparing the publish draft.",
                )
            }
        } else {
            item {
                SectionCard(title = "Selected clip") {
                    Text(selected.title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("${selected.provider} - ${selected.resolution}", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            item {
                SectionCard(title = "Caption") {
                    OutlinedTextField(
                        value = uiState.publishDraft.title,
                        onValueChange = { value -> onDraftChange { it.copy(title = value, mediumTitle = value, tiktokSubheading = value) } },
                        label = { Text("Title") },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = uiState.publishDraft.description,
                        onValueChange = { value -> onDraftChange { it.copy(description = value) } },
                        label = { Text("Description") },
                        minLines = 4,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 120.dp),
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = uiState.publishDraft.hashtagText(),
                        onValueChange = { value ->
                            val tags = value
                                .split(' ', ',', '\n', '\t')
                                .map { it.trim().removePrefix("#") }
                                .filter { it.isNotBlank() }
                                .distinct()
                            onDraftChange { it.copy(hashtags = tags) }
                        },
                        label = { Text("Hashtags") },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = uiState.publishDraft.category,
                        onValueChange = { value -> onDraftChange { it.copy(category = value.ifBlank { DEFAULT_YOUTUBE_CATEGORY }) } },
                        label = { Text("YouTube category") },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Button(
                            onClick = { shareVideo(context, selected, uiState.publishDraft, null) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Share sheet")
                        }
                        OutlinedButton(
                            onClick = { copyCaption(context, uiState.publishDraft) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Copy caption")
                        }
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        OutlinedButton(
                            onClick = { shareVideo(context, selected, uiState.publishDraft, YOUTUBE_PACKAGE) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("YouTube")
                        }
                        OutlinedButton(
                            onClick = { shareVideo(context, selected, uiState.publishDraft, TIKTOK_PACKAGE) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("TikTok")
                        }
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        OutlinedButton(
                            onClick = { shareVideo(context, selected, uiState.publishDraft, INSTAGRAM_PACKAGE) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Instagram")
                        }
                        OutlinedButton(
                            onClick = { shareVideo(context, selected, uiState.publishDraft, FACEBOOK_PACKAGE) },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Facebook")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SettingsScreen(
    uiState: StudioUiState,
    storagePath: String,
    onSettingsChange: ((StudioSettings) -> StudioSettings) -> Unit,
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item {
            InfoCard(
                title = "Stored locally on this device",
                body = "Credentials and workspace state stay local. Generated videos are written under the app's Movies folder so Android can manage sharing cleanly.",
            )
        }
        item {
            SectionCard(title = "xAI / Grok") {
                SecretTextField(
                    value = uiState.settings.grokApiKey,
                    onValueChange = { value -> onSettingsChange { it.copy(grokApiKey = value) } },
                    label = "Grok API key",
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.grokChatModel,
                    onValueChange = { value -> onSettingsChange { it.copy(grokChatModel = value) } },
                    label = { Text("Chat model") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.grokVideoModel,
                    onValueChange = { value -> onSettingsChange { it.copy(grokVideoModel = value) } },
                    label = { Text("Video model") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        item {
            SectionCard(title = "OpenAI") {
                SecretTextField(
                    value = uiState.settings.openAiApiKey,
                    onValueChange = { value -> onSettingsChange { it.copy(openAiApiKey = value) } },
                    label = "OpenAI API key",
                )
                Spacer(modifier = Modifier.height(12.dp))
                SecretTextField(
                    value = uiState.settings.openAiAccessToken,
                    onValueChange = { value -> onSettingsChange { it.copy(openAiAccessToken = value) } },
                    label = "OpenAI access token",
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.openAiChatModel,
                    onValueChange = { value -> onSettingsChange { it.copy(openAiChatModel = value) } },
                    label = { Text("Chat model") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.openAiSoraModel,
                    onValueChange = { value -> onSettingsChange { it.copy(openAiSoraModel = value) } },
                    label = { Text("Sora model") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        item {
            SectionCard(title = "Seedance") {
                SecretTextField(
                    value = uiState.settings.seedanceApiKey,
                    onValueChange = { value -> onSettingsChange { it.copy(seedanceApiKey = value) } },
                    label = "Seedance API key",
                )
                Spacer(modifier = Modifier.height(12.dp))
                SecretTextField(
                    value = uiState.settings.seedanceOauthToken,
                    onValueChange = { value -> onSettingsChange { it.copy(seedanceOauthToken = value) } },
                    label = "Seedance OAuth token",
                )
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedTextField(
                    value = uiState.settings.seedanceModel,
                    onValueChange = { value -> onSettingsChange { it.copy(seedanceModel = value) } },
                    label = { Text("Seedance model") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        item {
            SectionCard(title = "Storage") {
                Text(storagePath, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        item {
            InfoCard(
                title = "Feature parity status",
                body = "Android-native now: prompt creation, OpenAI/xAI/Seedance requests, local playback, import, and basic stitch export. Still desktop-only: browser automation uploaders, AI Flow Trainer, and the advanced desktop stitch pipeline.",
            )
        }
    }
}

private enum class CardTone {
    Neutral,
    Error,
}

@Composable
private fun SectionCard(
    title: String,
    content: @Composable () -> Unit,
) {
    ElevatedCard {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(14.dp))
            content()
        }
    }
}

@Composable
private fun InfoCard(
    title: String,
    body: String,
    modifier: Modifier = Modifier,
    tone: CardTone = CardTone.Neutral,
    action: @Composable (() -> Unit)? = null,
) {
    val contentColor = when (tone) {
        CardTone.Neutral -> MaterialTheme.colorScheme.onSecondaryContainer
        CardTone.Error -> MaterialTheme.colorScheme.onErrorContainer
    }
    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(18.dp),
        ) {
            Text(title, color = contentColor, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(8.dp))
            Text(body, color = contentColor)
            if (action != null) {
                Spacer(modifier = Modifier.height(8.dp))
                action()
            }
        }
    }
}

@Composable
private fun <T> EnumChipRow(
    title: String,
    options: List<T>,
    selected: T,
    label: (T) -> String,
    onSelected: (T) -> Unit,
) {
    Text(title, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Spacer(modifier = Modifier.height(8.dp))
    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        items(options) { option ->
            FilterChip(
                selected = option == selected,
                onClick = { onSelected(option) },
                label = { Text(label(option)) },
            )
        }
    }
}

@Composable
private fun StringChipRow(
    title: String,
    options: List<String>,
    selected: String,
    label: (String) -> String = { it },
    onSelected: (String) -> Unit,
) {
    EnumChipRow(
        title = title,
        options = options,
        selected = selected,
        label = label,
        onSelected = onSelected,
    )
}

@Composable
private fun SecretTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
) {
    var revealed by rememberSaveable { mutableStateOf(false) }
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        visualTransformation = if (revealed) androidx.compose.ui.text.input.VisualTransformation.None else PasswordVisualTransformation(),
        modifier = Modifier.fillMaxWidth(),
        trailingIcon = {
            TextButton(onClick = { revealed = !revealed }) {
                Text(if (revealed) "Hide" else "Show")
            }
        },
    )
}

@Composable
private fun VideoPreview(video: GeneratedVideo) {
    val file = File(video.filePath)
    if (!file.exists()) {
        InfoCard(
            title = "Video file missing",
            body = "The saved library entry still exists, but the actual file is no longer present on this device.",
            tone = CardTone.Error,
        )
        return
    }
    key(video.filePath) {
        AndroidView(
            factory = { context ->
                VideoView(context).apply {
                    setMediaController(MediaController(context).also { controller ->
                        controller.setAnchorView(this)
                    })
                    setVideoURI(Uri.fromFile(file))
                    setOnPreparedListener { mediaPlayer ->
                        mediaPlayer.isLooping = true
                        start()
                    }
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(240.dp),
        )
    }
}

@Composable
private fun VideoThumbnail(
    video: GeneratedVideo,
    modifier: Modifier = Modifier,
) {
    val file = File(video.filePath)
    val thumbnail by produceState<Bitmap?>(initialValue = null, key1 = video.filePath) {
        value = withContext(Dispatchers.IO) {
            if (!file.exists()) {
                null
            } else {
                runCatching {
                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(file.absolutePath)
                        retriever.frameAtTime
                    } finally {
                        retriever.release()
                    }
                }.getOrNull()
            }
        }
    }

    if (thumbnail != null) {
        Image(
            bitmap = thumbnail!!.asImageBitmap(),
            contentDescription = "${video.title} thumbnail",
            contentScale = ContentScale.Crop,
            modifier = modifier,
        )
    } else {
        Box(
            modifier = modifier,
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "Preview unavailable",
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun StudioTheme(content: @Composable () -> Unit) {
    val colorScheme = androidx.compose.material3.darkColorScheme(
        primary = Color(0xFF28D7F7),
        onPrimary = Color(0xFF062534),
        secondary = Color(0xFF7CE7F4),
        background = Color(0xFF07131D),
        surface = Color(0xFF0E2230),
        surfaceVariant = Color(0xFF143243),
        onSurface = Color(0xFFF5FBFF),
        onSurfaceVariant = Color(0xFFA8C5D7),
        secondaryContainer = Color(0xFF123244),
        onSecondaryContainer = Color(0xFFE8FAFF),
        primaryContainer = Color(0xFF1A4258),
        onPrimaryContainer = Color(0xFFE8FAFF),
        errorContainer = Color(0xFF5C1F28),
        onErrorContainer = Color(0xFFFDEBEC),
    )
    MaterialTheme(colorScheme = colorScheme, content = content)
}

private fun openVideo(context: Context, video: GeneratedVideo) {
    val file = File(video.filePath)
    if (!file.exists()) {
        return
    }
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val intent = Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(uri, "video/*")
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    runCatching {
        context.startActivity(intent)
    }
}

private fun shareVideo(
    context: Context,
    video: GeneratedVideo,
    metadata: SocialMetadata,
    packageName: String?,
) {
    val file = File(video.filePath)
    if (!file.exists()) {
        return
    }
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val caption = buildString {
        if (metadata.title.isNotBlank()) {
            append(metadata.title)
        }
        if (metadata.description.isNotBlank()) {
            if (isNotBlank()) append("\n\n")
            append(metadata.description)
        }
        if (metadata.hashtags.isNotEmpty()) {
            if (isNotBlank()) append("\n\n")
            append(metadata.hashtagText())
        }
    }
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "video/*"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_TITLE, metadata.title.ifBlank { video.title })
        putExtra(Intent.EXTRA_TEXT, caption)
        clipData = ClipData.newRawUri(video.title, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        if (!packageName.isNullOrBlank()) {
            setPackage(packageName)
            context.grantUriPermission(packageName, uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
    try {
        if (packageName.isNullOrBlank()) {
            context.startActivity(Intent.createChooser(intent, "Share video"))
        } else {
            context.startActivity(intent)
        }
    } catch (_: ActivityNotFoundException) {
        context.startActivity(Intent.createChooser(intent.setPackage(null), "Share video"))
    }
}

private fun copyCaption(context: Context, metadata: SocialMetadata) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager ?: return
    val text = buildString {
        append(metadata.title)
        if (metadata.description.isNotBlank()) {
            append("\n\n")
            append(metadata.description)
        }
        if (metadata.hashtags.isNotEmpty()) {
            append("\n\n")
            append(metadata.hashtagText())
        }
    }.trim()
    clipboard.setPrimaryClip(ClipData.newPlainText("publish caption", text))
}

private const val YOUTUBE_PACKAGE = "com.google.android.youtube"
private const val TIKTOK_PACKAGE = "com.zhiliaoapp.musically"
private const val INSTAGRAM_PACKAGE = "com.instagram.android"
private const val FACEBOOK_PACKAGE = "com.facebook.katana"
