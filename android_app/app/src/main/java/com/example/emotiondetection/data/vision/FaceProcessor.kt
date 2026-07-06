package com.example.emotiondetection.data.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.math.max
import kotlin.math.min

/**
 * Handles face detection and face-related image processing
 */
class FaceProcessor {
    private val faceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.15f) // Slightly larger for better performance
            .enableTracking()
            .build()
        FaceDetection.getClient(options)
    }
    
    companion object {
        private const val TAG = "FaceProcessor"
    }
      suspend fun detectFaces(bitmap: Bitmap): List<Face> = suspendCoroutine { continuation ->
        try {
            if (bitmap.isRecycled) {
                Log.e(TAG, "Cannot detect faces on recycled bitmap")
                continuation.resume(emptyList())
                return@suspendCoroutine
            }
            
            val image = InputImage.fromBitmap(bitmap, 0)
            
            faceDetector.process(image)
                .addOnSuccessListener { faces ->
                    Log.d(TAG, "Found ${faces.size} face(s)")
                    continuation.resume(faces)
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed", e)
                    continuation.resume(emptyList())
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up face detection", e)
            continuation.resume(emptyList())
        }
    }
      fun extractLargestFace(bitmap: Bitmap, faces: List<Face>): Bitmap? {
        try {
            if (bitmap.isRecycled) {
                Log.e(TAG, "Cannot extract face from recycled bitmap")
                return null
            }
            
            val largestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                ?: return null
                
            return extractFaceFromBitmap(bitmap, largestFace.boundingBox)
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting largest face", e)
            return null
        }
    }
    
    private fun extractFaceFromBitmap(bitmap: Bitmap, boundingBox: Rect): Bitmap? {
        return try {
            // Add some padding around the face
            val padding = 20
            val left = max(0, boundingBox.left - padding)
            val top = max(0, boundingBox.top - padding)
            val right = min(bitmap.width, boundingBox.right + padding)
            val bottom = min(bitmap.height, boundingBox.bottom + padding)

            val width = right - left
            val height = bottom - top

            if (width <= 0 || height <= 0) {
                Log.e(TAG, "Invalid face dimensions: ${width}x${height}")
                return null
            }

            Log.d(TAG, "Extracting face: ${left},${top} ${width}x${height} from ${bitmap.width}x${bitmap.height}")

            Bitmap.createBitmap(bitmap, left, top, width, height)
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting face: ${e.message}", e)
            null
        }
    }
    
    fun createVisualizationBitmap(originalBitmap: Bitmap, faces: List<Face>): Bitmap {
        val mutableBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        
        // Combined paint for both rectangle and text to reduce object creation
        val paint = Paint().apply {
            color = android.graphics.Color.GREEN
            isAntiAlias = true
        }

        // Draw bounding boxes around detected faces
        faces.forEach { face ->
            // Draw rectangle
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 5f
            canvas.drawRect(face.boundingBox, paint)
            
            // Draw text
            paint.style = Paint.Style.FILL
            paint.textSize = 40f
            canvas.drawText(
                "Face",
                face.boundingBox.left.toFloat(),
                face.boundingBox.top.toFloat() - 10,
                paint
            )
        }

        return mutableBitmap
    }
    
    fun close() {
        faceDetector.close()
    }
}
