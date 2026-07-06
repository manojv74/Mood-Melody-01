package com.example.emotiondetection.data.processing

import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.scale
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * Handles image preprocessing and conversion for ML model input
 */
class ImageProcessor {
    
    // Pre-allocated buffers for better memory performance
    private var cachedByteBuffer: ByteBuffer? = null
    private var cachedIntArray: IntArray? = null
    
    companion object {
        private const val TAG = "ImageProcessor"
        private const val INPUT_WIDTH = 224
        private const val INPUT_HEIGHT = 224
        private const val INPUT_CHANNELS = 3
        private val INPUT_MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val INPUT_STD = floatArrayOf(0.5f, 0.5f, 0.5f)
    }
      fun preprocessForModel(bitmap: Bitmap): Bitmap? {
        return try {
            if (bitmap.isRecycled) {
                Log.e(TAG, "Cannot preprocess recycled bitmap")
                return null
            }
            
            Log.d(TAG, "Preprocessing face image:")
            Log.d(TAG, "  Target: ${INPUT_WIDTH}x${INPUT_HEIGHT}x${INPUT_CHANNELS}")
            Log.d(TAG, "  Original face: ${bitmap.width}x${bitmap.height}")

            // Center crop to square (same as Python code)
            val minDimension = minOf(bitmap.width, bitmap.height)
            val xOffset = (bitmap.width - minDimension) / 2
            val yOffset = (bitmap.height - minDimension) / 2

            val squareBitmap = Bitmap.createBitmap(
                bitmap, xOffset, yOffset, minDimension, minDimension
            )
            // Scale to required size using high-quality filtering
            val scaledBitmap = if (minDimension == INPUT_WIDTH && minDimension == INPUT_HEIGHT) {
                // Already correct size, just use the square bitmap
                squareBitmap
            } else {
                // Use createScaledBitmap with filter=true for better quality
                Bitmap.createScaledBitmap(squareBitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
            }

            // Convert to RGB format
            val finalBitmap = if (scaledBitmap.config != Bitmap.Config.ARGB_8888) {
                scaledBitmap.copy(Bitmap.Config.ARGB_8888, false)
            } else {
                scaledBitmap
            }

            // Cleanup intermediate bitmaps
            safeBitmapRecycle(if (squareBitmap != finalBitmap) squareBitmap else null)
            safeBitmapRecycle(if (scaledBitmap != finalBitmap) scaledBitmap else null)

            Log.d(TAG, "Face preprocessing complete: ${finalBitmap.width}x${finalBitmap.height}")
            finalBitmap

        } catch (e: Exception) {
            Log.e(TAG, "ERROR in face preprocessing: ${e.message}", e)
            null
        }
    }    fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer? {
        return try {
            if (bitmap.isRecycled) {
                Log.e(TAG, "Cannot convert recycled bitmap to ByteBuffer")
                return null
            }
            
            // Always create fresh ByteBuffer to avoid corruption issues
            val bufferSize = 4 * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS
            val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
            byteBuffer.order(ByteOrder.nativeOrder())

            // Reuse int array if possible
            val pixelCount = INPUT_WIDTH * INPUT_HEIGHT
            val intValues = cachedIntArray?.takeIf { it.size == pixelCount }
                ?: IntArray(pixelCount).also { cachedIntArray = it }
                
            bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

            Log.d(TAG, "Converting face to ByteBuffer (NCHW format):")
            Log.d(TAG, "  Data type: FLOAT32")
            Log.d(TAG, "  Normalization: mean=${INPUT_MEAN.contentToString()}, std=${INPUT_STD.contentToString()}")
            Log.d(TAG, "  Format: NCHW [1, $INPUT_CHANNELS, $INPUT_HEIGHT, $INPUT_WIDTH]")

            // Process pixels with exact same normalization as Python code
            // Python: (image/255 - mean) / std
            // NCHW format: [batch, channels, height, width]

            // Extract and normalize all pixel values
            val normalizedPixels = Array(INPUT_CHANNELS) { Array(INPUT_HEIGHT) { FloatArray(INPUT_WIDTH) } }

            for (i in 0 until INPUT_HEIGHT) {
                for (j in 0 until INPUT_WIDTH) {
                    val pixelValue = intValues[i * INPUT_WIDTH + j]

                    // Extract RGB values
                    val r = (pixelValue shr 16) and 0xFF
                    val g = (pixelValue shr 8) and 0xFF
                    val b = pixelValue and 0xFF

                    // Normalize exactly like Python: (pixel/255 - mean) / std
                    normalizedPixels[0][i][j] = (r.toFloat() / 255.0f - INPUT_MEAN[0]) / INPUT_STD[0] // R channel
                    normalizedPixels[1][i][j] = (g.toFloat() / 255.0f - INPUT_MEAN[1]) / INPUT_STD[1] // G channel
                    normalizedPixels[2][i][j] = (b.toFloat() / 255.0f - INPUT_MEAN[2]) / INPUT_STD[2] // B channel
                }
            }

            // Write to buffer in NCHW order: all R values, then all G values, then all B values
            for (c in 0 until INPUT_CHANNELS) {
                for (i in 0 until INPUT_HEIGHT) {
                    for (j in 0 until INPUT_WIDTH) {
                        byteBuffer.putFloat(normalizedPixels[c][i][j])
                    }
                }
            }

            byteBuffer

        } catch (e: Exception) {
            Log.e(TAG, "ERROR converting face bitmap to ByteBuffer: ${e.message}", e)
            null
        }
    }
    
    fun applySoftmax(input: FloatArray): FloatArray {
        // Apply softmax function (same as Python implementation)
        val maxVal = input.maxOrNull() ?: 0f
        var sumExp = 0f
        val result = FloatArray(input.size)
        
        // Single pass: calculate exp values and sum simultaneously
        for (i in input.indices) {
            val expValue = exp((input[i] - maxVal).toDouble()).toFloat()
            result[i] = expValue
            sumExp += expValue
        }
        
        // Normalize by sum
        for (i in result.indices) {
            result[i] /= sumExp
        }
        
        return result
    }
    private fun safeBitmapRecycle(bitmap: Bitmap?) {
        if (bitmap != null && !bitmap.isRecycled) {
            bitmap.recycle()
        }
    }
      fun cleanup() {
        // Don't cache ByteBuffer anymore to prevent corruption
        cachedByteBuffer = null
        cachedIntArray = null
    }
}
