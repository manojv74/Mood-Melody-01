package com.example.emotiondetection.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.graphics.withSave

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00C853.toInt() // green
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    @Volatile
    private var boxes: List<RectF> = emptyList()

    fun setBoxes(newBoxes: List<RectF>) {
        boxes = newBoxes
        postInvalidateOnAnimation()
    }

    fun clear() {
        boxes = emptyList()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (boxes.isEmpty()) return
        canvas.withSave {
            for (rect in boxes) {
                canvas.drawRect(rect, boxPaint)
            }
        }
    }
}


