package com.example.emotiondetection.ui

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.emotiondetection.domain.model.EmotionDetectionResult
import com.example.emotiondetection.domain.repository.EmotionDetectionRepository
import kotlinx.coroutines.launch

class MainViewModel(
    private val emotionDetectionRepository: EmotionDetectionRepository = EmotionDetectionRepository()
) : ViewModel() {

    var state by mutableStateOf(MainScreenState())
        private set

    // Track consecutive failures for recovery logic
    private var consecutiveFailures = 0

    companion object {
        private const val TAG = "MainViewModel"
        private const val MAX_CONSECUTIVE_FAILURES = 2
    }
    
    /**
     * Initialize the model early to improve first detection performance
     */
    fun initializeModel(context: Context) {
        viewModelScope.launch {
            try {
                val success = emotionDetectionRepository.initializeModel(context)
                if (!success) {
                    Log.w(TAG, "Failed to pre-initialize model")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error pre-initializing model", e)
            }
        }
    }
    
    // Helper methods to reduce redundancy
    private fun updateStateWithError(message: String, exception: Exception? = null) {
        if (exception != null) {
            Log.e(TAG, "ERROR: $message", exception)
        }
        state = state.copy(
            lastResult = if (exception != null) "ERROR: $message - ${exception.message}" else message,
            isLoading = false,
            lastConfidence = null
        )
    }

    private fun updateStateWithLoading(message: String) {
        state = state.copy(
            lastResult = message,
            isLoading = true,
            lastConfidence = null
        )
    }    
    
    private fun updateStateWithSuccess(result: String, confidence: Float? = null, detectedImage: Bitmap? = null) {
        state = state.copy(
            lastResult = result,
            lastConfidence = confidence,
            isLoading = false,
            detectedImage = detectedImage ?: state.detectedImage
        )
    }    
    
    private fun safeBitmapRecycle(bitmap: Bitmap?) {
        try {
            if (bitmap != null && !bitmap.isRecycled) {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error recycling bitmap: ${e.message}")
        }
    }
    
    fun onEvent(event: MainScreenEvent) {
        when (event) {
            is MainScreenEvent.ResetDetection -> resetDetection()
            is MainScreenEvent.StartMonitoring,
            is MainScreenEvent.SelectFromGallery,
            is MainScreenEvent.SelectMusic -> {} // Handled in Activity
        }
    }    
    
    fun updateMusicStatus(isGenerated: Boolean?, emotion: String?) {
        state = state.copy(
            musicIsGenerated = isGenerated,
            musicEmotion = emotion ?: ""
        )
    }

    fun updateMonitoring(isMonitoring: Boolean) {
        state = state.copy(isMonitoring = isMonitoring)
    }
    
    /**
     * Continuous monitoring path: process a frame without showing loading states
     * or replacing the displayed bitmap. Suppresses 'No faces detected' updates
     * to avoid UI flicker; only updates when we have a positive result.
     */
    fun handleContinuousFrame(bitmap: Bitmap?, context: Context) {
        viewModelScope.launch {
            try {
                when (val result = emotionDetectionRepository.processImage(bitmap, context)) {
                    is EmotionDetectionResult.Success -> {
                        // Update only key fields; do not set isLoading or replace image
                        state = state.copy(
                            lastResult = result.emotion,
                            lastConfidence = result.confidence,
                            isLoading = false
                        )
                    }
                    is EmotionDetectionResult.NoFacesDetected -> {
                        // Suppress updating UI to avoid flicker; keep previous state
                    }
                    is EmotionDetectionResult.Error -> {
                        // Suppress frequent error messages during live monitoring
                    }
                    is EmotionDetectionResult.Loading -> {
                        // Ignore loading in continuous path
                    }
                }
            } catch (e: Exception) {
                // Suppress exceptions in continuous path
            }
        }
    }

    fun handleImageResult(bitmap: Bitmap?, context: Context) {
        // Don't recycle current bitmap yet - keep it until we successfully process the new one
        updateStateWithLoading("Processing image...")
        
        viewModelScope.launch {
            val result = emotionDetectionRepository.processImage(bitmap, context)
              when (result) {
                is EmotionDetectionResult.Success -> {
                    // Reset failure counter on success
                    consecutiveFailures = 0
                    // Only recycle old bitmap after successful processing
                    val oldBitmap = state.detectedImage
                    updateStateWithSuccess(
                        result = result.emotion,
                        confidence = result.confidence,
                        detectedImage = result.visualizationBitmap
                    )
                    // Safe to recycle old bitmap now
                    safeBitmapRecycle(oldBitmap)
                }
                is EmotionDetectionResult.Error -> {
                    consecutiveFailures++
                    Log.w(TAG, "Processing failed (consecutive failures: $consecutiveFailures)")
                    
                    // Force reload after multiple consecutive failures
                    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                        Log.w(TAG, "Too many consecutive failures, forcing component reload")
                        try {
                            emotionDetectionRepository.forceReload()
                            consecutiveFailures = 0 // Reset after forced reload
                            updateStateWithError("Multiple failures detected - components reloaded. Please try again.")
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to force reload", e)
                            updateStateWithError("Critical error: ${result.message}", result.exception)
                        }
                    } else {
                        updateStateWithError(result.message, result.exception)
                    }
                    // Keep the old image on error, don't recycle it
                }
                is EmotionDetectionResult.Loading -> {
                    updateStateWithLoading(result.message)
                }
                is EmotionDetectionResult.NoFacesDetected -> {
                    // Reset failure counter on successful processing (even if no faces)
                    consecutiveFailures = 0
                    // Only recycle old bitmap after successful processing
                    val oldBitmap = state.detectedImage
                    updateStateWithSuccess("No faces detected", detectedImage = result.originalBitmap)
                    // Safe to recycle old bitmap now
                    safeBitmapRecycle(oldBitmap)
                }
            }
        }
    }    
    
    private fun resetDetection() {
        safeBitmapRecycle(state.detectedImage)
        consecutiveFailures = 0 // Reset failure counter on manual reset
        state = state.copy(
            detectedImage = null,
            lastResult = "",
            lastConfidence = null,
            isLoading = false
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        safeBitmapRecycle(state.detectedImage)
        emotionDetectionRepository.cleanup()
    }
}