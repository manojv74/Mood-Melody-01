package com.example.emotiondetection

/**
 * Shared constants for both unit and instrumented tests.
 * This eliminates duplication across test files.
 */
object TestConstants {
    
    // Model configuration constants
    const val MODEL_INPUT_WIDTH = 224
    const val MODEL_INPUT_HEIGHT = 224
    const val MODEL_INPUT_CHANNELS = 3
    val MODEL_MEAN_VALUES = floatArrayOf(0.5f, 0.5f, 0.5f)
    val MODEL_STD_VALUES = floatArrayOf(0.5f, 0.5f, 0.5f)
    
    // Emotion labels array (must match MainViewModel.EMOTION_LABELS)
    val EMOTION_LABELS = arrayOf(
        "Neutral", "Happiness", "Surprise", "Sadness",
        "Anger", "Disgust", "Fear", "Contempt"
    )
    
    // Test tolerance for floating point comparisons
    const val FLOAT_TOLERANCE = 0.001f
    
    // Test package name
    const val APP_PACKAGE_NAME = "com.example.emotiondetection"
    
    // Model file name
    const val MODEL_FILE_NAME = "metadata.tflite"
}
