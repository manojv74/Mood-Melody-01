package com.example.emotiondetection.data.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.Path

/**
 * Manages TensorFlow Lite model loading and inference
 */
class TensorFlowModelManager {
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    companion object {
        private const val TAG = "TensorFlowModelManager"
        private const val MODEL_FILE = "metadata.tflite"
        private val EMOTION_LABELS = arrayOf(
            "Neutral", "Happiness", "Surprise", "Sadness",
            "Anger", "Disgust", "Fear", "Contempt"
        )
        private const val SERIALIZATION_DIR_NAME = "gpu_delegate_cache"
    }
    
    /**
     * Generate a unique model token for serialization
     */
    private fun generateModelToken(context: Context): String {
        return try {
            // Read model file to generate fingerprint
            val modelData = context.assets.open(MODEL_FILE).use { inputStream ->
                inputStream.readBytes()
            }
            
            // Generate MD5 hash as model token
            val digest = MessageDigest.getInstance("MD5")
            val hashBytes = digest.digest(modelData)
            hashBytes.joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to generate model token, using default: ${e.message}")
            "default_model_token"
        }
    }        
    
    /**
     * Get serialization directory for GPU delegate caching
     */
    private fun getSerializationDir(context: Context): java.io.File {
        val serializationPath = Path(context.codeCacheDir.absolutePath, SERIALIZATION_DIR_NAME)
        if (!serializationPath.exists()) {
            serializationPath.createDirectories()
        }
        return serializationPath.toFile()
    }

    fun loadModel(context: Context): Boolean {
        return try {
            Log.d(TAG, "=== LOADING MODEL ===")
            
            // Load model using Kotlin-friendly approach
            val modelBuffer = context.assets.openFd(MODEL_FILE).use { assetFileDescriptor ->
                assetFileDescriptor.createInputStream().use { inputStream ->
                    val fileChannel = inputStream.channel
                    fileChannel.map(
                        java.nio.channels.FileChannel.MapMode.READ_ONLY,
                        assetFileDescriptor.startOffset,
                        assetFileDescriptor.declaredLength
                    )
                }
            }
            
            // Create interpreter with GPU delegate if available
            val options = Interpreter.Options()
            options.setNumThreads(4)
            
            // Check if GPU delegate is available and compatible
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "GPU delegate is supported on this device")
                try {
                    // Enable serialization for faster subsequent loads
                    val serializationDir = getSerializationDir(context)
                    val modelToken = generateModelToken(context)
                    
                    Log.d(TAG, "GPU serialization enabled:")
                    Log.d(TAG, "  Serialization dir: ${serializationDir.absolutePath}")
                    Log.d(TAG, "  Model token: $modelToken")
                    
                    val delegateOptions = compatList.bestOptionsForThisDevice
                        .setSerializationParams(serializationDir.absolutePath, modelToken)
                    
                    gpuDelegate = GpuDelegate(delegateOptions)
                    options.addDelegate(gpuDelegate)
                    Log.d(TAG, "GPU delegate with serialization added successfully")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to add GPU delegate with serialization, trying without: ${e.message}")
                    try {
                        // Fallback to GPU without serialization
                        gpuDelegate?.close()
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        gpuDelegate = GpuDelegate(delegateOptions)
                        options.addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU delegate added without serialization")
                    } catch (fallbackException: Exception) {
                        Log.w(TAG, "Failed to add GPU delegate, falling back to CPU: ${fallbackException.message}")
                        try {
                            gpuDelegate?.close()
                        } catch (closeException: Exception) {
                            Log.w(TAG, "Error closing failed GPU delegate: ${closeException.message}")
                        }
                        gpuDelegate = null
                    }
                }
            } else {
                Log.d(TAG, "GPU delegate not supported on this device, using CPU")
            }
            
            interpreter = Interpreter(modelBuffer, options)            // Verify model configuration
            val inputTensor = interpreter!!.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val inputDataType = inputTensor.dataType()
            Log.d(TAG, "Model loaded successfully:")
            Log.d(TAG, "  Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "  Input data type: $inputDataType")
            Log.d(TAG, "  Labels: ${EMOTION_LABELS.contentToString()}")

            // Get output tensor info
            val outputTensor = interpreter!!.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            val outputDataType = outputTensor.dataType()

            Log.d(TAG, "Output tensor:")
            Log.d(TAG, "  Shape: ${outputShape.contentToString()}")
            Log.d(TAG, "  Data type: $outputDataType")
            Log.d(TAG, "  Expected classes: ${EMOTION_LABELS.size}")            
            Log.d(TAG, "=== MODEL LOADING COMPLETE ===")

            // Warm up the model with a dummy inference to reduce first-run latency
            warmUpModel()

            true

        } catch (e: Exception) {
            Log.e(TAG, "ERROR loading model: ${e.message}", e)
            false
        }
    }    
    
    fun runInference(inputBuffer: ByteBuffer): FloatArray? {
        return try {
            if (interpreter == null) {
                Log.e(TAG, "ERROR: Interpreter not initialized")
                return null
            }
            
            // Additional health check for consecutive calls
            if (!isInterpreterHealthy()) {
                Log.e(TAG, "ERROR: Interpreter is not in healthy state")
                return null
            }

            // Ensure buffer is rewound and in correct position
            inputBuffer.rewind()

            // Prepare output
            val outputArray = Array(1) { FloatArray(EMOTION_LABELS.size) }

            Log.d(TAG, "Running emotion inference...")
            val startTime = System.currentTimeMillis()

            interpreter!!.run(inputBuffer, outputArray)

            val inferenceTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Emotion inference completed in ${inferenceTime}ms")

            outputArray[0]

        } catch (e: Exception) {
            Log.e(TAG, "ERROR during inference: ${e.message}", e)
              // If GPU delegate causes issues, try to recover by falling back to CPU
            if (gpuDelegate != null && e.message?.contains("GPU", ignoreCase = true) == true) {
                Log.w(TAG, "GPU inference failed, attempting to recover with CPU-only mode")
                try {
                    // Close current interpreter and delegate
                    interpreter?.close()
                    gpuDelegate?.close()
                    gpuDelegate = null
                    interpreter = null
                    Log.w(TAG, "Interpreter reset due to GPU issues. Model needs to be reloaded.")
                } catch (recoveryException: Exception) {
                    Log.e(TAG, "Failed to recover from GPU error: ${recoveryException.message}")
                }
            }
            
            null
        }
    }
    
    fun getEmotionLabels(): Array<String> = EMOTION_LABELS
    fun needsReload(): Boolean = interpreter == null
    
    /**
     * Force a clean reload of the model (useful for recovery from errors)
     */
    fun forceReload() {
        Log.i(TAG, "Forcing model reload...")
        close()
    }

    /**
     * Check if the interpreter is in a healthy state for inference
     */
    fun isInterpreterHealthy(): Boolean {
        return try {
            if (interpreter == null) {
                return false
            }
            
            // Try to access interpreter state - this will throw if interpreter is closed/corrupted
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            
            // Basic validation that tensors are accessible
            inputTensor.shape() != null && outputTensor.shape() != null
            
        } catch (e: Exception) {
            Log.w(TAG, "Interpreter health check failed: ${e.message}")
            false
        }
    }
    
    private fun warmUpModel() {        
        try {
            Log.d(TAG, "Warming up model with dummy inference...")
            // Create dummy input buffer with correct size
            val dummyBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
            dummyBuffer.order(ByteOrder.nativeOrder())
            
            // Fill with dummy normalized data (0.0f)
            repeat(224 * 224 * 3) {
                dummyBuffer.putFloat(0.0f)
            }
            dummyBuffer.rewind()
            
            // Run dummy inference
            val startTime = System.currentTimeMillis()
            runInference(dummyBuffer)
            val warmupTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Model warm-up completed in ${warmupTime}ms")
        } catch (e: Exception) {
            Log.w(TAG, "Model warm-up failed, but continuing: ${e.message}")
        }
    }

    fun close() {
        try {
            interpreter?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing interpreter: ${e.message}")
        } finally {
            interpreter = null
        }
        
        try {
            gpuDelegate?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing GPU delegate: ${e.message}")
        } finally {
            gpuDelegate = null
        }
    }
}