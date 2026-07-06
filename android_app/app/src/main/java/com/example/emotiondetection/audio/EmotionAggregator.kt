package com.example.emotiondetection.audio

/**
 * Aggregates emotion detections over a sliding time window and computes a stable winner.
 */
class EmotionAggregator(
    private val windowMs: Long = 10_000L,
    private val minValidSamples: Int = 7,
    private val confidenceThreshold: Float = 0.55f,
    private val hysteresisMargin: Float = 0.12f // 12%
) {
    private data class Sample(val label: String, val confidence: Float, val timestampMs: Long)
    private val samples = ArrayDeque<Sample>()
    private var lastWinner: String? = null

    fun add(label: String, confidence: Float, nowMs: Long = System.currentTimeMillis()) {
        if (label.isBlank() || confidence < confidenceThreshold) return
        samples.addLast(Sample(label, confidence, nowMs))
        trim(nowMs)
    }

    private fun trim(nowMs: Long) {
        while (samples.isNotEmpty() && nowMs - samples.first().timestampMs > windowMs) {
            samples.removeFirst()
        }
    }

    fun currentWinner(nowMs: Long = System.currentTimeMillis()): String? {
        trim(nowMs)
        if (samples.size < minValidSamples) return lastWinner

        val counts = mutableMapOf<String, Float>()
        for (s in samples) {
            counts[s.label] = (counts[s.label] ?: 0f) + s.confidence
        }
        if (counts.isEmpty()) return lastWinner

        val sorted = counts.entries.sortedByDescending { it.value }
        val top = sorted[0]
        val second = if (sorted.size > 1) sorted[1] else null

        // Hysteresis: winner must beat previous by margin OR be same as last winner
        val winner = top.key
        val prev = lastWinner
        if (prev != null && prev != winner && second != null) {
            val prevScore = counts[prev] ?: 0f
            val marginOk = top.value >= prevScore * (1f + hysteresisMargin)
            if (!marginOk) return prev
        }

        lastWinner = winner
        return winner
    }

    fun reset() {
        samples.clear()
        lastWinner = null
    }
}


