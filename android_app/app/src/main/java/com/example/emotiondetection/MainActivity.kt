package com.example.emotiondetection

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.MediaPlayer
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import com.example.emotiondetection.ui.MainScreen
import com.example.emotiondetection.ui.MainScreenEvent
import com.example.emotiondetection.ui.MainViewModel
import com.example.emotiondetection.ui.theme.EmotionDetectionTheme
import com.example.emotiondetection.audio.MusicManager
import com.example.emotiondetection.audio.EmotionAggregator
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.BitmapFactory
import androidx.camera.view.PreviewView
import com.example.emotiondetection.ui.FaceOverlayView
import androidx.camera.core.ExperimentalGetImage
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.Locale

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
        // TODO: set this to your deployed server (use https in production)
        private const val SERVER_BASE_URL = "http://10.0.2.2:5000" // emulator -> host machine mapping; replace with your server URL
    }
    
    private val viewModel: MainViewModel by viewModels()
    private val supportedEmotionLabels = setOf(
        "Neutral",
        "Happiness",
        "Surprise",
        "Sadness",
        "Anger",
        "Disgust",
        "Fear",
        "Contempt"
    )
    private var lastRecognizedEmotion: String? = null
    
    // Track if we should open camera after permission granted
    private var shouldLaunchCameraAfterPermission = false
    
    // MediaPlayer for playing music
    private var mediaPlayer: MediaPlayer? = null
    
    // Default/generative music manager
    private lateinit var musicManager: MusicManager
    private val aggregatorConfidenceThreshold = 0.55f
    private lateinit var emotionAggregator: EmotionAggregator
    private val cadenceHandler by lazy { android.os.Handler(mainLooper) }
    // Stored in uptime millis, because Handler.postAtTime expects uptime values
    private var nextBoundaryMs: Long? = null
    private var useFixedCadence: Boolean = true
    
    // CameraX monitoring
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var isMonitoring: Boolean = false
    private var previewView: PreviewView? = null
    private var faceOverlayView: FaceOverlayView? = null
    private val mainHandler by lazy { android.os.Handler(mainLooper) }
    private val faceDetector by lazy {
        com.google.mlkit.vision.face.FaceDetection.getClient(
            com.google.mlkit.vision.face.FaceDetectorOptions.Builder()
                .setPerformanceMode(com.google.mlkit.vision.face.FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .enableTracking()
                .build()
        )
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->          if (isGranted) {
            // Only launch camera if requested from Capture Image button
            // In new flow, permission gates monitoring
            if (shouldLaunchCameraAfterPermission) startMonitoring()
        } else {
            showToast(R.string.camera_permission_required, Toast.LENGTH_LONG)
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            handleCameraResult(result.data)
        } else {
            showToast(R.string.image_capture_cancelled)
        }
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            handleGalleryResult(result.data)
        } else {
            showToast(R.string.image_capture_cancelled)
        }
    }

    private val audioFileLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            handleAudioFileResult(result.data)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize model early to improve first detection performance
        viewModel.initializeModel(this)
        // OkHttp client used for background requests to the music generation server
        val httpClient = OkHttpClient()

        musicManager = MusicManager(
            context = this,
            defaultWindowMs = 60_000L,
            onRequestGeneratedTrack = { emotionLabel, respondWithUri ->
                // Run network call off the main thread
                Executors.newSingleThreadExecutor().submit {
                    try {
                        val payload = JSONObject()
                        // Use 'default' user_id for now; customize per user if needed
                        payload.put("user_id", "default")
                        // Keep the same capitalization used by the Android code (server normalizes)
                        payload.put("mood", emotionLabel.lowercase(Locale.getDefault()))
                        // Provide a confidence estimate; we don't have exact value here so use a reasonable default
                        payload.put("confidence", 0.8)

                        val mediaType = "application/json; charset=utf-8".toMediaTypeOrNull()
                        val body = payload.toString().toRequestBody(mediaType)
                        val req = Request.Builder()
                            .url("${SERVER_BASE_URL}/generate")
                            .post(body)
                            .build()

                        httpClient.newCall(req).execute().use { resp ->
                            if (!resp.isSuccessful) {
                                runOnUiThread { respondWithUri(null) }
                                return@use
                            }
                            val text = resp.body?.string()
                            if (text == null) {
                                runOnUiThread { respondWithUri(null) }
                                return@use
                            }
                            val json = JSONObject(text)
                            val musicUrl = json.optString("music_url", null)
                            if (musicUrl != null) {
                                val uri = Uri.parse(musicUrl)
                                runOnUiThread { respondWithUri(uri) }
                            } else {
                                runOnUiThread { respondWithUri(null) }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to request generated track: ${e.message}")
                        runOnUiThread { respondWithUri(null) }
                    }
                }
            },
            onPlaybackStatusChanged = { isGenerated, emotion ->
                viewModel.updateMusicStatus(isGenerated, emotion)
            }
        )
    emotionAggregator = EmotionAggregator(confidenceThreshold = aggregatorConfidenceThreshold)
        
        setContent {
            EmotionDetectionTheme {
                MainScreenContent()
            }
        }
    }

    @Composable
    private fun MainScreenContent() {
        val state = viewModel.state
        var checkCameraPermissionOnStart by remember { mutableStateOf(true) }
        
        // Check camera permission on first composition without launching camera
        LaunchedEffect(checkCameraPermissionOnStart) {
            if (checkCameraPermissionOnStart) {
                requestCameraPermissionOnly()
                checkCameraPermissionOnStart = false
            }
        }
        
        MainScreen(
            state = state,
            onEvent = { event ->
                when (event) {
                    is MainScreenEvent.StartMonitoring -> launchMonitoring()
                    is MainScreenEvent.SelectFromGallery -> launchGallery()
                    is MainScreenEvent.SelectMusic -> launchAudioFilePicker()
                    is MainScreenEvent.ResetDetection -> {
                        stopMonitoring()
                        musicManager.stopPlayback()
                        clearOverlay()
                        emotionAggregator.reset()
                        viewModel.onEvent(event)
                    }
                    else -> viewModel.onEvent(event)
                }
            }
        )
        
        // Aggregate detections continuously; cadence scheduler decides when to act
        LaunchedEffect(state.lastResult, state.lastConfidence, state.isLoading) {
            if (!state.isLoading && state.lastResult.isNotBlank()) {
                val conf = state.lastConfidence ?: 0f
                if (supportedEmotionLabels.contains(state.lastResult) && conf >= aggregatorConfidenceThreshold) {
                    lastRecognizedEmotion = state.lastResult
                }
                emotionAggregator.add(state.lastResult, conf)
            }
        }
    }
    
    // Helper method to reduce toast redundancy
    private fun showToast(messageResId: Int, duration: Int = Toast.LENGTH_SHORT) {
        Toast.makeText(this, getString(messageResId), duration).show()
    }    
    
    // Helper method to reduce error handling redundancy
    private fun handleError(action: String, exception: Exception, messageResId: Int) {
        Log.e(TAG, "Error $action: ${exception.message}")
        showToast(messageResId)
    }

    // Helper method to check camera permission (reduces duplication)
    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    // Helper method for intent resolution pattern
    private fun launchIntentIfAvailable(intent: Intent, launcher: (Intent) -> Unit, noAppMessageResId: Int) {
        if (intent.resolveActivity(packageManager) != null) {
            launcher(intent)
        } else {
            showToast(noAppMessageResId)
        }
    }

    // Request permission without launching camera
    private fun requestCameraPermissionOnly() {
        if (!hasCameraPermission()) {
            shouldLaunchCameraAfterPermission = false // Explicitly set to false to not launch camera
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // This function checks permission and launches monitoring when button is pressed
    private fun launchMonitoring() {
        if (!hasCameraPermission()) {
            shouldLaunchCameraAfterPermission = true // Set flag to launch monitoring after permission
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            viewModel.updateMonitoring(true)
            startMonitoring()
        }
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    @OptIn(ExperimentalGetImage::class)
    private fun startMonitoring() {
        if (isMonitoring) return
        isMonitoring = true
        cameraExecutor = Executors.newSingleThreadExecutor()
        startFixedCadenceIfNeeded()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                val previewUseCase = androidx.camera.core.Preview.Builder().build().also { preview ->
                    preview.setSurfaceProvider(previewView?.surfaceProvider)
                }
                imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        var lastProcessed = 0L
                        analysis.setAnalyzer(cameraExecutor!!) { imageProxy ->
                            try {
                                // Throttle processing (e.g., every 750ms)
                                val now = System.currentTimeMillis()
                                if (now - lastProcessed >= 1000) {
                                    lastProcessed = now
                                    val bitmap = imageProxy.toBitmapCorrected()
                                    // Use lightweight continuous path so UI doesn’t flicker while still sharing the
                                    // same ML pipeline output semantics as gallery capture.
                                    viewModel.handleContinuousFrame(bitmap, this@MainActivity)
                                    val mediaImage = imageProxy.image
                                    if (mediaImage != null) {
                                        val rotation = imageProxy.imageInfo.rotationDegrees
                                        val inputImage = com.google.mlkit.vision.common.InputImage.fromMediaImage(mediaImage, rotation)
                                        faceDetector.process(inputImage)
                                            .addOnSuccessListener { faces ->
                                                updateOverlayBoxes(faces, imageProxy)
                                            }
                                            .addOnFailureListener {
                                                clearOverlay()
                                            }
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "Analyzer error: ${e.message}")
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                cameraProvider.bindToLifecycle(this, cameraSelector, previewUseCase, imageAnalysis)
            } catch (e: Exception) {
                Log.e(TAG, "Error starting monitoring: ${e.message}")
                isMonitoring = false
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopMonitoring() {
        if (!isMonitoring) return
        try {
            val cameraProvider = ProcessCameraProvider.getInstance(this).get()
            cameraProvider.unbindAll()
        } catch (_: Exception) { }
        imageAnalysis = null
        cameraExecutor?.shutdownNow()
        cameraExecutor = null
        isMonitoring = false
        cancelCadence()
        viewModel.updateMonitoring(false)
    }

    private fun cancelCadence() {
        nextBoundaryMs = null
        cadenceHandler.removeCallbacksAndMessages(null)
    }

    private fun launchGallery() {
        try {
            val pickPhotoIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            launchIntentIfAvailable(
                pickPhotoIntent,
                { galleryLauncher.launch(it) },
                R.string.no_gallery_app_found
            )        } catch (e: Exception) {
            handleError("launching gallery", e, R.string.unable_to_select_image)
        }
    }
      
    private fun handleCameraResult(data: Intent?) {
        try {
            val imageBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                data?.extras?.getParcelable("data", Bitmap::class.java)
            } else {
                @Suppress("DEPRECATION")
                data?.extras?.getParcelable("data")
            }
            processImageResult(imageBitmap)   
        } catch (e: Exception) {
            handleError("handling camera result", e, R.string.error_processing_image)
        }
    }    
    
    private fun handleGalleryResult(data: Intent?) {
        try {
            val imageUri = data?.data
            val imageBitmap = if (imageUri != null) {
                // Use modern ImageDecoder (API 28+ required)
                val source = ImageDecoder.createSource(contentResolver, imageUri)
                ImageDecoder.decodeBitmap(source)
            } else {
                showToast(R.string.unable_to_select_image)
                return
            }
            processImageResult(imageBitmap)
        } catch (e: Exception) {
            handleError("handling gallery result", e, R.string.error_processing_image)
        }
    }

    // Extract common image processing logic
    private fun processImageResult(imageBitmap: Bitmap?) {
        viewModel.handleImageResult(imageBitmap, this)
    }

    // Convert ImageProxy to Bitmap with rotation/mirroring correction for front camera
    private fun ImageProxy.toBitmapCorrected(): Bitmap? {
        return try {
            val nv21 = yuv420888ToNv21()
            val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 95, out)
            val imageBytes = out.toByteArray()
            out.close()
            val raw = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size) ?: return null
            val matrix = android.graphics.Matrix().apply {
                val rotation = imageInfo.rotationDegrees.toFloat()
                if (rotation != 0f) {
                    postRotate(rotation)
                }
                // Mirror horizontally because we use the front camera
                postScale(-1f, 1f, raw.width / 2f, raw.height / 2f)
            }
            val corrected = android.graphics.Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
            if (corrected != raw && !raw.isRecycled) {
                raw.recycle()
            }
            corrected
        } catch (e: Exception) {
            Log.e(TAG, "toBitmap error: ${e.message}")
            null
        }
    }

    private fun ImageProxy.yuv420888ToNv21(): ByteArray {
        val width = width
        val height = height
        val ySize = width * height
        val uvSize = width * height / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer.duplicate().apply { rewind() }
        val uBuffer = uPlane.buffer.duplicate().apply { rewind() }
        val vBuffer = vPlane.buffer.duplicate().apply { rewind() }

        val yBytes = ByteArray(yBuffer.remaining())
        yBuffer.get(yBytes)

        val uBytes = ByteArray(uBuffer.remaining())
        uBuffer.get(uBytes)

        val vBytes = ByteArray(vBuffer.remaining())
        vBuffer.get(vBytes)

        var outputPos = 0
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        for (row in 0 until height) {
            var inputOffset = row * yRowStride
            for (col in 0 until width) {
                nv21[outputPos++] = yBytes[inputOffset]
                inputOffset += yPixelStride
            }
        }

        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        var uvPos = ySize
        for (row in 0 until height / 2) {
            var uOffset = row * uRowStride
            var vOffset = row * vRowStride
            for (col in 0 until width / 2) {
                nv21[uvPos++] = vBytes[vOffset]
                nv21[uvPos++] = uBytes[uOffset]
                uOffset += uPixelStride
                vOffset += vPixelStride
            }
        }

        return nv21
    }
    private fun launchAudioFilePicker() {
        try {
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                type = "audio/*"
                addCategory(Intent.CATEGORY_OPENABLE)
                // Filter for WAV and MP3 files
                putExtra(
                    Intent.EXTRA_MIME_TYPES,
                    arrayOf("audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3")
                )
            }
            launchIntentIfAvailable(
                intent,
                { audioFileLauncher.launch(it) },
                R.string.no_audio_app_found
            )
        } catch (e: Exception) {
            handleError("launching audio file picker", e, R.string.unable_to_select_music)
        }
    }

    private fun handleAudioFileResult(data: Intent?) {
        try {
            val audioUri = data?.data
            if (audioUri != null) {
                playAudio(audioUri)
            } else {
                showToast(R.string.unable_to_select_music)
            }
        } catch (e: Exception) {
            handleError("handling audio file result", e, R.string.error_playing_music)
        }
    }

    private fun playAudio(uri: Uri) {
        try {
            // Stop and release previous MediaPlayer if exists
            mediaPlayer?.release()
            mediaPlayer = null

            // Create and configure new MediaPlayer
            mediaPlayer = MediaPlayer().apply {
                setDataSource(applicationContext, uri)
                prepare()
                start()
                
                // Release MediaPlayer when playback completes
                setOnCompletionListener {
                    it.release()
                    mediaPlayer = null
                }
                
                // Handle errors
                setOnErrorListener { _, what, extra ->
                    Log.e(TAG, "MediaPlayer error: what=$what, extra=$extra")
                    release()
                    mediaPlayer = null
                    showToast(R.string.error_playing_music)
                    true
                }
            }
        } catch (e: Exception) {
            handleError("playing audio", e, R.string.error_playing_music)
            mediaPlayer?.release()
            mediaPlayer = null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release MediaPlayer when activity is destroyed
        mediaPlayer?.release()
        mediaPlayer = null
        if (this::musicManager.isInitialized) {
            musicManager.release()
        }
        stopMonitoring()
    }

    // Fixed cadence: boundaries at 10s, 70s, 130s ... from start
    private fun startFixedCadenceIfNeeded() {
        if (!useFixedCadence) return
        if (nextBoundaryMs != null) return
        val now = SystemClock.uptimeMillis()
        val firstBoundary = now + 10_000L
        nextBoundaryMs = firstBoundary
        scheduleBoundary(firstBoundary)
    }

    private fun scheduleBoundary(boundaryTime: Long) {
        cadenceHandler.postAtTime({ onBoundary(boundaryTime) }, boundaryTime)
        // Also schedule detection window end at boundary + 10s
        val detectEnd = boundaryTime + 10_000L
        cadenceHandler.postAtTime({ onDetectWindowEnd(detectEnd) }, detectEnd)
    }

    private fun onBoundary(boundaryTime: Long) {
        // Winner from previous 10s window ends at boundaryTime
        val winner = emotionAggregator.currentWinner()
        val fallbackFromState = viewModel.state.lastResult.takeIf { supportedEmotionLabels.contains(it) }
        val emotionToPlay = when {
            !winner.isNullOrBlank() && supportedEmotionLabels.contains(winner) -> winner
            fallbackFromState != null -> fallbackFromState
            else -> lastRecognizedEmotion
        }
        if (!emotionToPlay.isNullOrBlank()) {
            musicManager.onEmotionDetected(emotionToPlay)
        } else {
            Log.d(TAG, "No valid emotion detected for playback window ending at $boundaryTime")
        }
        // Start next detection window [boundary, boundary+10]
        emotionAggregator.reset()
        // Schedule next boundary +60s
        val next = boundaryTime + 60_000L
        nextBoundaryMs = next
        scheduleBoundary(next)
    }

    private fun onDetectWindowEnd(detectEndTime: Long) {
        // Detection window [boundary, boundary+10] winner
        val winner = emotionAggregator.currentWinner()
        if (!winner.isNullOrBlank()) {
            // TODO: Replace with backend call; when URI is ready, supply to music manager
            // For now we simulate no URI; MusicManager will continue default until available
            // musicManager.setPendingGeneratedUri(generatedUri)
        }
    }

    fun bindPreview(preview: PreviewView, overlay: FaceOverlayView) {
        previewView = preview
        faceOverlayView = overlay
    }

    private fun updateOverlayBoxes(
        faces: List<com.google.mlkit.vision.face.Face>,
        imageProxy: ImageProxy
    ) {
        val overlay = faceOverlayView ?: return
        val preview = previewView ?: return
        if (faces.isEmpty()) {
            clearOverlay()
            return
        }
        val rotation = imageProxy.imageInfo.rotationDegrees
        val imageWidth = if (rotation == 90 || rotation == 270) imageProxy.height else imageProxy.width
        val imageHeight = if (rotation == 90 || rotation == 270) imageProxy.width else imageProxy.height

        val viewWidth = preview.width.toFloat().coerceAtLeast(1f)
        val viewHeight = preview.height.toFloat().coerceAtLeast(1f)

        // FILL_CENTER mapping
        val scale = maxOf(viewWidth / imageWidth, viewHeight / imageHeight)
        val dx = (viewWidth - imageWidth * scale) / 2f
        val dy = (viewHeight - imageHeight * scale) / 2f

        val isFrontCamera = true

        val rects = faces.map { face ->
            val b = face.boundingBox
            val x = b.left.toFloat()
            val y = b.top.toFloat()
            val w = b.width().toFloat()
            val h = b.height().toFloat()

            val mappedXImage = if (isFrontCamera) (imageWidth - (x + w)) else x

            val left = mappedXImage * scale + dx
            val top = y * scale + dy
            val right = left + w * scale
            val bottom = top + h * scale
            android.graphics.RectF(left, top, right, bottom)
        }
        mainHandler.post { overlay.setBoxes(rects) }
    }

    private fun clearOverlay() {
        faceOverlayView?.let { view ->
            mainHandler.post { view.clear() }
        }
    }
}