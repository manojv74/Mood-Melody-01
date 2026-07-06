package com.example.emotiondetection

import com.example.emotiondetection.ui.MainScreenState
import org.junit.Test
import org.junit.Assert.*
import kotlin.math.exp

/**
 * Unit tests for emotion detection app logic.
 * These tests run on the JVM and test pure logic functions
 * without requiring Android context.
 */
class EmotionDetectionUnitTest {

    @Test
    fun testAddition() {
        assertEquals(4, 2 + 2)
    }

    @Test
    fun testMainScreenStateInitialization() {
        val state = MainScreenState()
        
        assertNull("Initial detected image should be null", state.detectedImage)
        assertEquals("Initial result should be empty", "", state.lastResult)
        assertNull("Initial confidence should be null", state.lastConfidence)
        assertFalse("Initial loading state should be false", state.isLoading)
    }

    @Test
    fun testMainScreenStateUpdate() {
        val initialState = MainScreenState()
        val updatedState = initialState.copy(
            lastResult = "Happiness",
            lastConfidence = 0.85f,
            isLoading = false
        )
        
        assertEquals("Result should be updated", "Happiness", updatedState.lastResult)
        assertEquals("Confidence should be updated", 0.85f, updatedState.lastConfidence!!, TestConstants.FLOAT_TOLERANCE)
        assertFalse("Loading should be false", updatedState.isLoading)
    }
    
    @Test
    fun testEmotionLabels() {
        assertEquals("Should have 8 emotion labels", 8, TestConstants.EMOTION_LABELS.size)
        
        // Test that all labels are properly formatted
        for (label in TestConstants.EMOTION_LABELS) {
            assertFalse("Label should not be empty", label.isEmpty())
            assertTrue("Label should start with uppercase", label[0].isUpperCase())
            assertTrue("Label should contain only letters", label.all { it.isLetter() })
        }
        
        // Test specific labels
        assertTrue("Should contain Neutral", TestConstants.EMOTION_LABELS.contains("Neutral"))
        assertTrue("Should contain Happiness", TestConstants.EMOTION_LABELS.contains("Happiness"))
        assertTrue("Should contain Anger", TestConstants.EMOTION_LABELS.contains("Anger"))
    }

    @Test
    fun testSoftmaxFunction() {
        // Test softmax implementation
        val input = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val result = applySoftmax(input)
        
        // Check that probabilities sum to 1
        val sum = result.sum()
        assertEquals("Softmax probabilities should sum to 1", 1.0f, sum, TestConstants.FLOAT_TOLERANCE)
        
        // Check that all values are positive
        for (value in result) {
            assertTrue("All softmax values should be positive", value > 0)
            assertTrue("All softmax values should be <= 1", value <= 1.0f)
        }
        
        // Check that the highest input has the highest probability
        val maxInputIndex = input.indices.maxByOrNull { input[it] } ?: 0
        val maxResultIndex = result.indices.maxByOrNull { result[it] } ?: 0
        assertEquals("Highest input should have highest probability", maxInputIndex, maxResultIndex)
    }

    @Test
    fun testSoftmaxWithZeros() {
        val input = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        val result = applySoftmax(input)
        
        val sum = result.sum()
        assertEquals("Softmax with zeros should sum to 1", 1.0f, sum, TestConstants.FLOAT_TOLERANCE)
        
        // All values should be equal (1/4 = 0.25)
        for (value in result) {
            assertEquals("All values should be equal", 0.25f, value, TestConstants.FLOAT_TOLERANCE)
        }
    }

    @Test
    fun testSoftmaxWithNegativeValues() {
        val input = floatArrayOf(-1.0f, -2.0f, -3.0f, 0.0f)
        val result = applySoftmax(input)
        
        val sum = result.sum()
        assertEquals("Softmax with negative values should sum to 1", 1.0f, sum, TestConstants.FLOAT_TOLERANCE)
        
        // The zero value should have the highest probability
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: 0
        assertEquals("Zero input should have highest probability", 3, maxIndex)
    }
    
    @Test
    fun testModelConfiguration() {
        // Test model configuration constants using shared values
        assertTrue("Input width should be positive", TestConstants.MODEL_INPUT_WIDTH > 0)
        assertTrue("Input height should be positive", TestConstants.MODEL_INPUT_HEIGHT > 0)
        assertTrue("Input channels should be 3", TestConstants.MODEL_INPUT_CHANNELS == 3)
        
        assertEquals("Mean should have 3 values", 3, TestConstants.MODEL_MEAN_VALUES.size)
        assertEquals("Std should have 3 values", 3, TestConstants.MODEL_STD_VALUES.size)
        
        for (i in 0 until 3) {
            assertTrue("Mean values should be reasonable", TestConstants.MODEL_MEAN_VALUES[i] in 0.0f..1.0f)
            assertTrue("Std values should be positive", TestConstants.MODEL_STD_VALUES[i] > 0.0f)
        }
    }

    @Test
    fun testPixelNormalization() {
        // Test pixel normalization logic (0-255 to normalized values)
        val pixelValue = 128 // Mid-range pixel value
        val mean = 0.5f
        val std = 0.5f
        
        val normalizedValue = (pixelValue.toFloat() / 255.0f - mean) / std
        
        // For pixel=128, normalized should be (128/255 - 0.5) / 0.5 = 0.003921...
        val expected = (128.0f / 255.0f - 0.5f) / 0.5f
        assertEquals("Pixel normalization should be correct", expected, normalizedValue, TestConstants.FLOAT_TOLERANCE)
    }

    @Test
    fun testConfidenceValidation() {
        // Test confidence value validation
        val testConfidences = arrayOf(0.0f, 0.5f, 0.99f, 1.0f, 1.1f, -0.1f)
        
        for (confidence in testConfidences) {
            val validatedConfidence = confidence.coerceIn(0f, 1f)
            assertTrue("Confidence should be >= 0", validatedConfidence >= 0f)
            assertTrue("Confidence should be <= 1", validatedConfidence <= 1f)
        }
    }

    @Test
    fun testArrayOperations() {
        // Test array operations used in emotion detection
        val testArray = floatArrayOf(0.1f, 0.8f, 0.3f, 0.2f)
        
        // Test finding max index
        val maxIndex = testArray.indices.maxByOrNull { testArray[it] } ?: 0
        assertEquals("Max index should be 1", 1, maxIndex)
        assertEquals("Max value should be 0.8", 0.8f, testArray[maxIndex], TestConstants.FLOAT_TOLERANCE)
        
        // Test array sum
        val sum = testArray.sum()
        assertEquals("Sum should be 1.4", 1.4f, sum, TestConstants.FLOAT_TOLERANCE)
        
        // Test array max value
        val maxValue = testArray.maxOrNull() ?: 0f
        assertEquals("Max value should be 0.8", 0.8f, maxValue, TestConstants.FLOAT_TOLERANCE)
    }

    /**
     * Helper function to apply softmax (mirrors the MainViewModel implementation)
     */
    private fun applySoftmax(input: FloatArray): FloatArray {
        val maxVal = input.maxOrNull() ?: 0f
        val expValues = input.map { exp((it - maxVal).toDouble()).toFloat() }
        val sumExp = expValues.sum()
        return expValues.map { it / sumExp }.toFloatArray()
    }
}