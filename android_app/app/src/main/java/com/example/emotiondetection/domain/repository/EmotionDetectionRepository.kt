package com.example.emotiondetection.domain.repository

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.example.emotiondetection.data.ml.TensorFlowModelManager
import com.example.emotiondetection.data.processing.ImageProcessor
import com.example.emotiondetection.data.vision.FaceProcessor
import com.example.emotiondetection.domain.model.EmotionDetectionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Repository that orchestrates the emotion detection pipeline
 * Optimized for performance with memory management and caching
 */
class EmotionDetectionRepository(
    private val modelManager: TensorFlowModelManager = TensorFlowModelManager(),
    private val faceProcessor: FaceProcessor = FaceProcessor(),
    private val imageProcessor: ImageProcessor = ImageProcessor()
) {
    
    // Cache for model loading state
    private var isModelInitialized = false
    
    companion object {
        private const val TAG = "EmotionDetectionRepository"
    }
    
    /**
     * Initialize the model ahead of time to avoid slow first detection
     */
    suspend fun initializeModel(context: Context): Boolean {
        return withContext(Dispatchers.Default) {
            try {
                if (!isModelInitialized) {
                    val modelLoadStart = System.currentTimeMillis()
                    if (modelManager.loadModel(context)) {
                        isModelInitialized = true
                        Log.d(TAG, "Model pre-initialized in ${System.currentTimeMillis() - modelLoadStart}ms")
                        true
                    } else {
                        Log.e(TAG, "Failed to pre-initialize model")
                        false
                    }
                } else {
                    true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error pre-initializing model", e)
                false
            }
        }
    }    
    
    suspend fun processImage(bitmap: Bitmap?, context: Context): EmotionDetectionResult {
        return withContext(Dispatchers.Default) {
            val totalStartTime = System.currentTimeMillis()
            try {
                if (bitmap == null) {
                    return@withContext EmotionDetectionResult.Error("Unable to capture image")
                }
                
                if (bitmap.isRecycled) {
                    Log.e(TAG, "Cannot process recycled bitmap")
                    return@withContext EmotionDetectionResult.Error("Image is no longer available")
                }

                Log.d(TAG, "Starting emotion detection pipeline...")                // Load model if not done (with caching) or if it needs reload
                // Also check interpreter health for consecutive captures
                if (!isModelInitialized || modelManager.needsReload() || !modelManager.isInterpreterHealthy()) {
                    val modelLoadStart = System.currentTimeMillis()
                    if (!modelManager.loadModel(context)) {
                        isModelInitialized = false // Reset flag on failure
                        return@withContext EmotionDetectionResult.Error("Cannot load model")
                    }
                    isModelInitialized = true
                    Log.d(TAG, "Model loaded/reloaded in ${System.currentTimeMillis() - modelLoadStart}ms")
                }// Detect faces
                val faceDetectStart = System.currentTimeMillis()
                val faces = faceProcessor.detectFaces(bitmap)
                Log.d(TAG, "Face detection completed in ${System.currentTimeMillis() - faceDetectStart}ms")
                
                if (faces.isEmpty()) {
                    Log.w(TAG, "No faces detected in image")
                    return@withContext EmotionDetectionResult.NoFacesDetected(bitmap)
                }

                Log.d(TAG, "Found ${faces.size} face(s)")                // Create visualization bitmap early to avoid holding multiple bitmaps
                val visualizationBitmap = faceProcessor.createVisualizationBitmap(bitmap, faces)

                // Extract largest face
                val faceBitmap = faceProcessor.extractLargestFace(bitmap, faces)
                if (faceBitmap == null) {
                    safeBitmapRecycle(visualizationBitmap)
                    return@withContext EmotionDetectionResult.Error("Failed to extract face")
                }

                // Preprocess face for model
                val processedFaceBitmap = imageProcessor.preprocessForModel(faceBitmap)
                if (processedFaceBitmap == null) {
                    safeBitmapRecycle(faceBitmap)
                    safeBitmapRecycle(visualizationBitmap)
                    return@withContext EmotionDetectionResult.Error("Face preprocessing failed")
                }

                // Convert to model input format
                val inputBuffer = imageProcessor.bitmapToByteBuffer(processedFaceBitmap)
                if (inputBuffer == null) {
                    safeBitmapRecycle(faceBitmap)
                    safeBitmapRecycle(processedFaceBitmap)
                    safeBitmapRecycle(visualizationBitmap)
                    return@withContext EmotionDetectionResult.Error("Failed to convert face image to model input format")
                }

                // Early cleanup to free memory sooner
                safeBitmapRecycle(faceBitmap)
                if (processedFaceBitmap != faceBitmap) {
                    safeBitmapRecycle(processedFaceBitmap)
                }                // Run inference
                val rawOutput = modelManager.runInference(inputBuffer)
                if (rawOutput == null) {
                    // Mark model as needing reload if inference fails
                    isModelInitialized = false
                    safeBitmapRecycle(visualizationBitmap)
                    return@withContext EmotionDetectionResult.Error("Model inference failed - will reload on next attempt")
                }

                // Process results with validation
                val softmaxOutput = imageProcessor.applySoftmax(rawOutput)
                val maxIndex = softmaxOutput.indices.maxByOrNull { softmaxOutput[it] } ?: 0
                val confidence = softmaxOutput[maxIndex].coerceIn(0f, 1f)
                val detectedEmotion = modelManager.getEmotionLabels()[maxIndex]                
                Log.d(TAG, "Emotion detected: $detectedEmotion (confidence: ${(confidence * 100).toInt()}%)")
                  val totalTime = System.currentTimeMillis() - totalStartTime
                Log.d(TAG, "Total emotion detection pipeline completed in ${totalTime}ms")

                // Force garbage collection between consecutive captures to prevent memory pressure
                System.gc()

                EmotionDetectionResult.Success(detectedEmotion, confidence, visualizationBitmap)} catch (e: Exception) {
                Log.e(TAG, "Error in processImage", e)
                // Reset model state on critical errors to ensure clean state for next attempt
                if (e.message?.contains("interpreter", ignoreCase = true) == true ||
                    e.message?.contains("tflite", ignoreCase = true) == true ||
                    e.message?.contains("gpu", ignoreCase = true) == true) {
                    Log.w(TAG, "Critical error detected, resetting model state")
                    isModelInitialized = false
                }
                EmotionDetectionResult.Error("Error processing image", e)
            }
        }
    }      
    
    private fun safeBitmapRecycle(bitmap: Bitmap?) {
        try {
            if (bitmap != null && !bitmap.isRecycled) {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error recycling bitmap: ${e.message}")
        }
    }    fun cleanup() {
        modelManager.close()
        faceProcessor.close()
        imageProcessor.cleanup()
        isModelInitialized = false
    }
    
    /**
     * Force a complete reload of all components (useful for recovery from errors)
     */
    fun forceReload() {
        Log.i(TAG, "Forcing complete component reload...")
        cleanup()
        modelManager.forceReload()
    }
}