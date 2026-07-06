package com.example.emotiondetection.domain.model

import android.graphics.Bitmap

/**
 * Represents the result of emotion detection processing
 */
sealed class EmotionDetectionResult {
    data class Success(
        val emotion: String,
        val confidence: Float,
        val visualizationBitmap: Bitmap
    ) : EmotionDetectionResult()
    
    data class Error(
        val message: String,
        val exception: Exception? = null
    ) : EmotionDetectionResult()
    
    data class Loading(val message: String) : EmotionDetectionResult()
    
    data class NoFacesDetected(val originalBitmap: Bitmap) : EmotionDetectionResult()
}
