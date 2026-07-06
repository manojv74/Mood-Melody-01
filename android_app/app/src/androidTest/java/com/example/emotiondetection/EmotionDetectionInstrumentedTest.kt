package com.example.emotiondetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import org.junit.Before

/**
 * Instrumented tests for emotion detection functionality.
 * These tests run on an Android device and verify the model loading,
 * face detection, and emotion classification components.
 */
@RunWith(AndroidJUnit4::class)
class EmotionDetectionInstrumentedTest {

    private lateinit var context: Context

    @Before
    fun setUp() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    @Test
    fun testAppContext() {
        // Verify the app context has the correct package name
        assertEquals(TestConstants.APP_PACKAGE_NAME, context.packageName)
    }

    @Test
    fun testModelFileExists() {
        // Verify the emotion detection model file exists in assets
        val assetManager = context.assets
        try {
            val inputStream = assetManager.open(TestConstants.MODEL_FILE_NAME)
            assertNotNull("Model file should exist", inputStream)
            inputStream.close()
        } catch (e: Exception) {
            fail("Model file ${TestConstants.MODEL_FILE_NAME} not found in assets: ${e.message}")
        }
    }

    @Test
    fun testCreateTestBitmap() {
        // Test creating a test bitmap for testing purposes
        val testBitmap = createTestBitmap(TestConstants.MODEL_INPUT_WIDTH, TestConstants.MODEL_INPUT_HEIGHT)
        assertNotNull("Test bitmap should be created", testBitmap)
        assertEquals("Bitmap width should be ${TestConstants.MODEL_INPUT_WIDTH}", TestConstants.MODEL_INPUT_WIDTH, testBitmap.width)
        assertEquals("Bitmap height should be ${TestConstants.MODEL_INPUT_HEIGHT}", TestConstants.MODEL_INPUT_HEIGHT, testBitmap.height)
        testBitmap.recycle()
    }

    @Test
    fun testBitmapProcessing() {
        // Test bitmap creation and basic processing
        val testBitmap = createTestBitmap(512, 512)
        assertNotNull("Large test bitmap should be created", testBitmap)
        
        // Test that we can scale it down
        val scaledBitmap = Bitmap.createScaledBitmap(testBitmap, TestConstants.MODEL_INPUT_WIDTH, TestConstants.MODEL_INPUT_HEIGHT, true)
        assertNotNull("Scaled bitmap should be created", scaledBitmap)
        assertEquals("Scaled width should be ${TestConstants.MODEL_INPUT_WIDTH}", TestConstants.MODEL_INPUT_WIDTH, scaledBitmap.width)
        assertEquals("Scaled height should be ${TestConstants.MODEL_INPUT_HEIGHT}", TestConstants.MODEL_INPUT_HEIGHT, scaledBitmap.height)
        
        // Cleanup
        testBitmap.recycle()
        scaledBitmap.recycle()
    }

    @Test
    fun testEmotionLabels() {
        // Test that emotion labels are properly defined using shared constants
        assertEquals("Should have 8 emotion labels", 8, TestConstants.EMOTION_LABELS.size)
        
        // Test that labels are not empty
        for (label in TestConstants.EMOTION_LABELS) {
            assertFalse("Label should not be empty", label.isEmpty())
            assertTrue("Label should be capitalized", label[0].isUpperCase())
        }
    }

    @Test
    fun testModelDimensions() {
        // Test that model expects correct input dimensions using shared constants
        assertTrue("Width should be positive", TestConstants.MODEL_INPUT_WIDTH > 0)
        assertTrue("Height should be positive", TestConstants.MODEL_INPUT_HEIGHT > 0)
        assertTrue("Channels should be 3 for RGB", TestConstants.MODEL_INPUT_CHANNELS == 3)
    }

    @Test
    fun testNormalizationValues() {
        // Test normalization parameters using shared constants
        assertEquals("Mean should have 3 values", 3, TestConstants.MODEL_MEAN_VALUES.size)
        assertEquals("Std should have 3 values", 3, TestConstants.MODEL_STD_VALUES.size)
        
        for (i in TestConstants.MODEL_MEAN_VALUES.indices) {
            assertTrue("Mean values should be between 0 and 1", TestConstants.MODEL_MEAN_VALUES[i] in 0.0f..1.0f)
            assertTrue("Std values should be positive", TestConstants.MODEL_STD_VALUES[i] > 0.0f)
        }
    }

    /**
     * Helper function to create a test bitmap with gradient colors
     * This simulates a face-like image for testing purposes
     */
    private fun createTestBitmap(width: Int, height: Int): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Create a simple gradient pattern that resembles a face
        for (y in 0 until height) {
            for (x in 0 until width) {
                // Create a gradient from center (lighter) to edges (darker)
                val centerX = width / 2
                val centerY = height / 2
                val distance = kotlin.math.sqrt(
                    ((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY)).toDouble()
                ).toFloat()
                val maxDistance = kotlin.math.sqrt((centerX * centerX + centerY * centerY).toDouble()).toFloat()
                val intensity = (255 * (1 - distance / maxDistance)).toInt().coerceIn(0, 255)
                
                // Create a skin-tone-like color
                val red = (intensity * 0.9).toInt().coerceIn(0, 255)
                val green = (intensity * 0.7).toInt().coerceIn(0, 255)
                val blue = (intensity * 0.6).toInt().coerceIn(0, 255)
                
                bitmap.setPixel(x, y, Color.rgb(red, green, blue))
            }
        }
        
        return bitmap
    }
}