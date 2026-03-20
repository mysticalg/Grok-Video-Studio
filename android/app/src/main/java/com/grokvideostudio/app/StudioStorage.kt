package com.grokvideostudio.app

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class StudioStorage(context: Context) {

    private val gson = Gson()
    private val preferences: SharedPreferences = createPreferences(context.applicationContext)
    private val videosType = object : TypeToken<List<GeneratedVideo>>() {}.type

    fun loadSettings(): StudioSettings {
        val raw = preferences.getString(KEY_SETTINGS, null) ?: return StudioSettings()
        return runCatching { gson.fromJson(raw, StudioSettings::class.java) }.getOrDefault(StudioSettings())
    }

    fun saveSettings(settings: StudioSettings) {
        preferences.edit(commit = true) {
            putString(KEY_SETTINGS, gson.toJson(settings))
        }
    }

    fun loadVideos(): List<GeneratedVideo> {
        val raw = preferences.getString(KEY_VIDEOS, null) ?: return emptyList()
        return runCatching { gson.fromJson<List<GeneratedVideo>>(raw, videosType) }.getOrDefault(emptyList())
    }

    fun saveVideos(videos: List<GeneratedVideo>) {
        preferences.edit(commit = true) {
            putString(KEY_VIDEOS, gson.toJson(videos))
        }
    }

    fun loadSelectedVideoId(): String? = preferences.getString(KEY_SELECTED_VIDEO_ID, null)

    fun saveSelectedVideoId(videoId: String?) {
        preferences.edit(commit = true) {
            if (videoId.isNullOrBlank()) {
                remove(KEY_SELECTED_VIDEO_ID)
            } else {
                putString(KEY_SELECTED_VIDEO_ID, videoId)
            }
        }
    }

    private fun createPreferences(context: Context): SharedPreferences {
        return runCatching {
            val masterKey = MasterKey.Builder(context)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            EncryptedSharedPreferences.create(
                context,
                PREFERENCES_NAME,
                masterKey,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
            )
        }.getOrElse {
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
        }
    }

    private companion object {
        const val PREFERENCES_NAME = "grok_video_studio_android"
        const val KEY_SETTINGS = "settings"
        const val KEY_VIDEOS = "videos"
        const val KEY_SELECTED_VIDEO_ID = "selected_video_id"
    }
}
