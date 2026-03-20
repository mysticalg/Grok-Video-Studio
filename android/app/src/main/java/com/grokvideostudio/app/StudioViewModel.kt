package com.grokvideostudio.app

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class StudioViewModel(application: Application) : AndroidViewModel(application) {

    private val storage = StudioStorage(application.applicationContext)
    private val repository = StudioRepository(application.applicationContext)
    private var settingsPersistJob: Job? = null

    private val _uiState = MutableStateFlow(
        createInitialState(
            settings = storage.loadSettings(),
            videos = storage.loadVideos().sortedByDescending { it.createdAtEpochMs },
            selectedVideoId = storage.loadSelectedVideoId(),
        ),
    )
    val uiState: StateFlow<StudioUiState> = _uiState.asStateFlow()

    fun updateSettings(transform: (StudioSettings) -> StudioSettings) {
        val updated = transform(_uiState.value.settings)
        _uiState.update { it.copy(settings = updated) }
        scheduleSettingsPersist(updated)
    }

    fun updatePublishDraft(transform: (SocialMetadata) -> SocialMetadata) {
        val current = _uiState.value
        val updatedDraft = transform(current.publishDraft)
        val selectedId = current.selectedVideoId
        val updatedVideos = if (selectedId == null) {
            current.videos
        } else {
            current.videos.map { video ->
                if (video.id == selectedId) {
                    video.copy(
                        title = updatedDraft.title.ifBlank { video.title },
                        description = updatedDraft.description,
                        hashtags = updatedDraft.hashtags,
                        category = updatedDraft.category.ifBlank { DEFAULT_YOUTUBE_CATEGORY },
                    )
                } else {
                    video
                }
            }
        }
        val updatedSettings = current.settings.copy(socialMetadata = updatedDraft)
        _uiState.update {
            it.copy(
                publishDraft = updatedDraft,
                videos = updatedVideos,
                settings = updatedSettings,
            )
        }
        storage.saveVideos(updatedVideos)
        scheduleSettingsPersist(updatedSettings)
    }

    fun selectVideo(videoId: String) {
        val selected = _uiState.value.videos.firstOrNull { it.id == videoId } ?: return
        _uiState.update {
            it.copy(
                selectedVideoId = selected.id,
                publishDraft = selected.toSocialMetadata(),
                errorMessage = null,
            )
        }
        storage.saveSelectedVideoId(selected.id)
    }

    fun toggleVideoForStitch(videoId: String) {
        _uiState.update { state ->
            val updatedSelection = state.stitchSelectionIds.toMutableSet().apply {
                if (!add(videoId)) {
                    remove(videoId)
                }
            }
            state.copy(stitchSelectionIds = updatedSelection)
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun generatePrompt() {
        val settings = _uiState.value.settings
        launchBusy("Generating prompt and social metadata...") {
            val result = repository.generatePrompt(settings)
            val updatedSettings = _uiState.value.settings.copy(
                manualPrompt = result.manualPrompt,
                socialMetadata = result.metadata,
            )
            _uiState.update {
                it.copy(
                    settings = updatedSettings,
                    publishDraft = result.metadata,
                    statusMessage = "Prompt ready for generation.",
                )
            }
            storage.saveSettings(updatedSettings)
            appendLog("Prompt generated for ${updatedSettings.videoProvider.label}.")
        }
    }

    fun generateVideo() {
        val settings = _uiState.value.settings
        launchBusy("Starting video generation...") {
            val video = repository.generateVideo(settings) { status ->
                _uiState.update { state -> state.copy(statusMessage = status) }
            }
            val updatedVideos = listOf(video) + _uiState.value.videos
            _uiState.update {
                it.copy(
                    videos = updatedVideos,
                    selectedVideoId = video.id,
                    publishDraft = video.toSocialMetadata(),
                    statusMessage = "Video saved to device.",
                )
            }
            storage.saveVideos(updatedVideos)
            storage.saveSelectedVideoId(video.id)
            appendLog("Saved ${video.title} to ${repository.mediaDirectory().absolutePath}.")
        }
    }

    fun importVideo(uri: Uri) {
        launchBusy("Importing video from device...") {
            val video = repository.importVideo(uri)
            val updatedVideos = listOf(video) + _uiState.value.videos
            _uiState.update {
                it.copy(
                    videos = updatedVideos,
                    selectedVideoId = video.id,
                    publishDraft = video.toSocialMetadata(),
                    statusMessage = "Video imported.",
                )
            }
            storage.saveVideos(updatedVideos)
            storage.saveSelectedVideoId(video.id)
            appendLog("Imported ${video.title}.")
        }
    }

    fun stitchSelectedVideos() {
        val current = _uiState.value
        val selectedVideos = current.videos.filter { it.id in current.stitchSelectionIds }
        if (selectedVideos.size < 2) {
            _uiState.update { it.copy(errorMessage = "Select at least two clips in Library before stitching.") }
            return
        }
        launchBusy("Starting stitch export...") {
            val stitchedVideo = repository.stitchVideos(selectedVideos) { status ->
                _uiState.update { state -> state.copy(statusMessage = status) }
            }
            val updatedVideos = listOf(stitchedVideo) + _uiState.value.videos
            _uiState.update {
                it.copy(
                    videos = updatedVideos,
                    selectedVideoId = stitchedVideo.id,
                    stitchSelectionIds = emptySet(),
                    publishDraft = stitchedVideo.toSocialMetadata(),
                    statusMessage = "Stitched video saved to device.",
                )
            }
            storage.saveVideos(updatedVideos)
            storage.saveSelectedVideoId(stitchedVideo.id)
            appendLog("Stitch export complete: ${stitchedVideo.title}.")
        }
    }

    fun removeSelectedVideo() {
        val current = _uiState.value
        val selectedId = current.selectedVideoId ?: return
        val selectedVideo = current.videos.firstOrNull { it.id == selectedId } ?: return
        runCatching {
            File(selectedVideo.filePath).takeIf { it.exists() }?.delete()
        }
        val updatedVideos = current.videos.filterNot { it.id == selectedId }
        val nextSelected = updatedVideos.firstOrNull()
        _uiState.update {
            it.copy(
                videos = updatedVideos,
                selectedVideoId = nextSelected?.id,
                stitchSelectionIds = current.stitchSelectionIds - selectedId,
                publishDraft = nextSelected?.toSocialMetadata() ?: current.settings.socialMetadata,
                statusMessage = if (updatedVideos.isEmpty()) "Library is empty." else "Video removed.",
            )
        }
        storage.saveVideos(updatedVideos)
        storage.saveSelectedVideoId(nextSelected?.id)
        appendLog("Removed ${selectedVideo.title} from the device library.")
    }

    fun mediaDirectoryPath(): String = repository.mediaDirectory().absolutePath

    private fun launchBusy(initialStatus: String, block: suspend () -> Unit) {
        _uiState.update { it.copy(isBusy = true, statusMessage = initialStatus, errorMessage = null) }
        appendLog(initialStatus)
        viewModelScope.launch {
            runCatching { block() }
                .onFailure { error ->
                    _uiState.update {
                        it.copy(
                            isBusy = false,
                            errorMessage = error.message ?: "Something went wrong.",
                            statusMessage = "Last action failed.",
                        )
                    }
                    appendLog("Error: ${error.message ?: "Unknown failure"}")
                }
                .onSuccess {
                    _uiState.update { it.copy(isBusy = false) }
                }
        }
    }

    private fun appendLog(message: String) {
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.UK).format(Date())
        _uiState.update { state ->
            state.copy(activityLog = listOf("[$timestamp] $message") + state.activityLog.take(23))
        }
    }

    private fun scheduleSettingsPersist(settings: StudioSettings) {
        settingsPersistJob?.cancel()
        settingsPersistJob = viewModelScope.launch(Dispatchers.IO) {
            delay(250)
            storage.saveSettings(settings)
        }
    }

    private fun createInitialState(
        settings: StudioSettings,
        videos: List<GeneratedVideo>,
        selectedVideoId: String?,
    ): StudioUiState {
        val actualSelectedId = selectedVideoId?.takeIf { candidate -> videos.any { it.id == candidate } }
            ?: videos.firstOrNull()?.id
        val selectedVideo = videos.firstOrNull { it.id == actualSelectedId }
        return StudioUiState(
            settings = settings,
            videos = videos,
            selectedVideoId = actualSelectedId,
            publishDraft = selectedVideo?.toSocialMetadata() ?: settings.socialMetadata,
            statusMessage = if (videos.isEmpty()) {
                "Ready to create videos on Android."
            } else {
                "Library loaded from device storage."
            },
        )
    }
}
