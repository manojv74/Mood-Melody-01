package com.example.emotiondetection.audio

import android.content.Context
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Manages music playback for detected emotions.
 *
 * Behavior:
 * - When a new emotion is detected, starts playing its default track.
 * - Plays the default track for a fixed "default window" (e.g., 60 seconds).
 * - If emotions change during the window, the latest emotion is queued and will switch only
 *   at the next window boundary.
 * - After the default window ends, this class will call an optional callback to request
 *   a generated track for the current (or queued) emotion. If no generated track is provided,
 *   it continues with the default behavior (play default for the queued emotion).
 *
 * NOTE: Default tracks are expected under the app assets directory:
 * assets/default_music/{neutral|happiness|surprise|sadness|anger|disgust|fear|contempt}.wav
 */
class MusicManager(
    private val context: Context,
    private val defaultWindowMs: Long = 60_000L,
    private val onRequestGeneratedTrack: ((emotionLabel: String, respondWithUri: (android.net.Uri?) -> Unit) -> Unit)? = null,
    private val onPlaybackStatusChanged: ((isGenerated: Boolean, emotionLabel: String) -> Unit)? = null
) {
    private var mediaPlayer: MediaPlayer? = null
    private val handler = Handler(Looper.getMainLooper())

    private var isInDefaultWindow: Boolean = false
    private var currentEmotion: String? = null
    private var queuedEmotion: String? = null
    private var pendingGeneratedUri: android.net.Uri? = null

    private val endOfDefaultWindowRunnable = Runnable {
        handleEndOfDefaultWindow()
    }

    /**
     * Supply a generated track URI to be used at the end of the current default window.
     * If called multiple times within a window, the latest URI will be used.
     */
    fun setPendingGeneratedUri(uri: android.net.Uri?) {
        pendingGeneratedUri = uri
    }

    private fun mapEmotionToAssetBase(emotion: String): String? {
        return when (emotion.trim()) {
            "Neutral" -> "neutral"
            "Happiness" -> "happiness"
            "Surprise" -> "surprise"
            "Sadness" -> "sadness"
            "Anger" -> "anger"
            "Disgust" -> "disgust"
            "Fear" -> "fear"
            "Contempt" -> "contempt"
            else -> null
        }
    }

    private fun openDefaultAssetFileDescriptor(baseName: String): android.content.res.AssetFileDescriptor? {
        val candidates = listOf(
            "default_music/$baseName.mp3",
            "default_music/$baseName.wav"
        )
        for (path in candidates) {
            try {
                return context.assets.openFd(path)
            } catch (_: Exception) {
                // try next
            }
        }
        return null
    }

    fun onEmotionDetected(emotionLabel: String) {
        if (emotionLabel.isBlank()) return

        if (!isInDefaultWindow || currentEmotion == null) {
            startDefaultForEmotion(emotionLabel)
        } else {
            queuedEmotion = emotionLabel
        }
    }

    private fun startDefaultForEmotion(emotionLabel: String) {
        stopPlayback()
        currentEmotion = emotionLabel
        queuedEmotion = null
        isInDefaultWindow = true
        onPlaybackStatusChanged?.invoke(false, emotionLabel)

        // Kick off generated track request at the START of the default window to allow ~1 minute for generation
        onRequestGeneratedTrack?.invoke(emotionLabel) { uri ->
            // Store for use at end of window
            pendingGeneratedUri = uri
        }

        val baseName = mapEmotionToAssetBase(emotionLabel)
        if (baseName == null) {
            Log.w(TAG, "No default asset mapping for emotion: $emotionLabel")
            return
        }

        try {
            val afd = openDefaultAssetFileDescriptor(baseName)
            if (afd == null) {
                Log.w(TAG, "Default asset not found for emotion: $emotionLabel")
                return
            }
            mediaPlayer = MediaPlayer().apply {
                setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                prepare()
                start()
                setOnCompletionListener {
                    // If the file is shorter than the window, restart until window ends
                    if (isInDefaultWindow) {
                        it.seekTo(0)
                        it.start()
                    } else {
                        it.release()
                        mediaPlayer = null
                    }
                }
                setOnErrorListener { mp, what, extra ->
                    Log.e(TAG, "MediaPlayer error in default playback: what=$what, extra=$extra")
                    mp.reset()
                    true
                }
            }
            handler.postDelayed(endOfDefaultWindowRunnable, defaultWindowMs)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start default track for '$emotionLabel': ${e.message}")
        }
    }

    private fun handleEndOfDefaultWindow() {
        isInDefaultWindow = false
        val latestQueuedEmotion = queuedEmotion
        val emotionAtBoundary = latestQueuedEmotion ?: currentEmotion

        if (emotionAtBoundary == null) {
            stopPlayback()
            return
        }

        // If we have a pending generated track ready within this window, switch to it now.
        val readyUri = pendingGeneratedUri
        pendingGeneratedUri = null
        if (readyUri != null) {
            playGenerated(readyUri, emotionAtBoundary)
            // After switching to generated, honor future emotion changes only at the next boundary.
            currentEmotion = emotionAtBoundary
            queuedEmotion = null
        } else {
            // Generated track not ready; switch to next emotion's default if queued, else continue same emotion.
            if (latestQueuedEmotion != null && latestQueuedEmotion != currentEmotion) {
                startDefaultForEmotion(latestQueuedEmotion)
            } else {
                startDefaultForEmotion(emotionAtBoundary)
            }
        }
    }

    private fun playGenerated(uri: android.net.Uri, emotionLabel: String) {
        stopPlayback()
        try {
            onPlaybackStatusChanged?.invoke(true, emotionLabel)
            mediaPlayer = MediaPlayer().apply {
                setDataSource(context, uri)
                prepare()
                start()
                setOnCompletionListener {
                    // When generated ends, start a new default window to keep the loop running.
                    it.release()
                    mediaPlayer = null
                    val nextEmotion = queuedEmotion ?: currentEmotion
                    queuedEmotion = null
                    if (!nextEmotion.isNullOrBlank()) {
                        startDefaultForEmotion(nextEmotion!!)
                    }
                }
                setOnErrorListener { mp, what, extra ->
                    Log.e(TAG, "MediaPlayer error in generated playback: what=$what, extra=$extra")
                    mp.reset()
                    true
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start generated track: ${e.message}")
        }
    }

    fun stopPlayback() {
        handler.removeCallbacks(endOfDefaultWindowRunnable)
        mediaPlayer?.release()
        mediaPlayer = null
        isInDefaultWindow = false
    }

    fun release() {
        stopPlayback()
        pendingGeneratedUri = null
        currentEmotion = null
        queuedEmotion = null
    }

    companion object {
        private const val TAG = "MusicManager"
    }
}


