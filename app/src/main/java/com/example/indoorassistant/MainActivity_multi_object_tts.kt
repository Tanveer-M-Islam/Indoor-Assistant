package com.example.indoorassistant

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Half
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView

import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import kotlin.math.max
import kotlin.math.min


class MainActivity :
    AppCompatActivity(),
    TextToSpeech.OnInitListener {


    // ============================================================
    // EXP-09F
    // FINAL CONTROLLED REAL-DEVICE BENCHMARK
    // ============================================================


    // ============================================================
    // MODEL CONFIGURATION
    // ============================================================

    private val MODEL_NAME =
        "EXP05_FP16.onnx"

    private val INPUT_SIZE =
        640

    private val NUM_CLASSES =
        16

    private val NUM_PREDICTIONS =
        8400

    private val OUTPUT_CHANNELS =
        20

    private val NMS_IOU_THRESHOLD =
        0.70f

    private val MAX_DETECTIONS =
        100


    // ============================================================
    // CLASS NAMES
    // ============================================================

    private val classNames =
        arrayOf(
            "bed",
            "sofa",
            "chair",
            "table",
            "lamp",
            "tv",
            "laptop",
            "wardrobe",
            "window",
            "door",
            "potted plant",
            "photo frame",
            "scissors",
            "knife",
            "cutter",
            "syringe"
        )


    // ============================================================
    // SAFETY CLASSES
    // ============================================================

    private val SAFETY_CLASS_IDS =
        setOf(
            12,     // scissors
            13,     // knife
            14,     // cutter
            15      // syringe
        )


    // ============================================================
    // FROZEN THRESHOLDS
    // ============================================================

    private val GENERAL_THRESHOLD =
        0.40f


    private fun thresholdForClass(
        classId: Int
    ): Float {

        return when (classId) {

            12 -> 0.40f

            13 -> 0.25f

            14 -> 0.50f

            15 -> 0.50f

            else ->
                GENERAL_THRESHOLD
        }
    }


    // ============================================================
    // TEMPORAL CONFIRMATION
    // ============================================================

    private val HISTORY_SIZE =
        3

    private val REQUIRED_CONFIRMATIONS =
        2

    private val detectionHistory =
        ArrayDeque<Set<Int>>()


    // ============================================================
    // TTS COOLDOWN
    // ============================================================

    private val SPEECH_COOLDOWN_MS =
        5000L

    private val lastSpokenTime =
        mutableMapOf<Int, Long>()


    // ============================================================
    // TTS
    // ============================================================

    private var textToSpeech:
            TextToSpeech? =
        null


    @Volatile
    private var ttsReady =
        false


    @Volatile
    private var ttsSpeaking =
        false


    private var speechCounter =
        0L



    // ============================================================
    // FP16 LOOKUP TABLE
    // ============================================================

    private val fp16Lookup =
        ShortArray(256) { value ->

            Half.toHalf(
                value / 255.0f
            )
        }


    // ============================================================
    // REUSABLE PREPROCESSING MEMORY
    // ============================================================

    private val pixelBuffer =
        IntArray(
            INPUT_SIZE *
                    INPUT_SIZE
        )


    private val inputByteBuffer =
        ByteBuffer
            .allocateDirect(
                3 *
                        INPUT_SIZE *
                        INPUT_SIZE *
                        2
            )
            .order(
                ByteOrder.nativeOrder()
            )


    private val inputShortBuffer =
        inputByteBuffer
            .asShortBuffer()


    private val modelBitmap =
        Bitmap.createBitmap(
            INPUT_SIZE,
            INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )


    private val modelCanvas =
        Canvas(
            modelBitmap
        )


    private val resizePaint =
        Paint(
            Paint.FILTER_BITMAP_FLAG
        )


    // ============================================================
    // REUSABLE OUTPUT BUFFER
    // ============================================================

    private val rawOutput =
        FloatArray(
            OUTPUT_CHANNELS *
                    NUM_PREDICTIONS
        )


    // ============================================================
    // ONNX
    // ============================================================

    private lateinit var ortEnvironment:
            OrtEnvironment


    private var ortSession:
            OrtSession? =
        null


    private var inputTensor:
            OnnxTensor? =
        null


    private lateinit var inputName:
            String


    // ============================================================
    // CAMERA/UI
    // ============================================================

    private lateinit var previewView:
            PreviewView


    private lateinit var overlayView:
            DetectionOverlay


    private lateinit var statsText:
            TextView


    private lateinit var cameraExecutor:
            ExecutorService


    // ============================================================
    // DATA CLASSES
    // ============================================================

    data class Detection(

        val classId: Int,

        val confidence: Float,

        val x1: Float,

        val y1: Float,

        val x2: Float,

        val y2: Float
    )


    data class FrameTransform(

        val originalWidth: Int,

        val originalHeight: Int,

        val scale: Float,

        val padX: Float,

        val padY: Float
    )


    // ============================================================
    // CAMERA PERMISSION
    // ============================================================

    private val cameraPermissionLauncher =

        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {

                initializeModelAndCamera()

            } else {

                statsText.text =
                    "Camera permission denied."
            }
        }


    // ============================================================
    // ON CREATE
    // ============================================================

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )


        createInterface()


        textToSpeech =
            TextToSpeech(
                this,
                this
            )


        cameraExecutor =
            Executors.newSingleThreadExecutor()


        if (

            ContextCompat.checkSelfPermission(

                this,

                Manifest.permission.CAMERA

            ) == PackageManager.PERMISSION_GRANTED

        ) {

            initializeModelAndCamera()

        } else {

            cameraPermissionLauncher.launch(
                Manifest.permission.CAMERA
            )
        }
    }


    // ============================================================
    // TTS INITIALIZATION
    // ============================================================

    override fun onInit(
        status: Int
    ) {

        if (
            status ==
            TextToSpeech.SUCCESS
        ) {

            val tts =
                textToSpeech
                    ?: return


            val languageResult =
                tts.setLanguage(
                    Locale.US
                )


            if (

                languageResult ==
                TextToSpeech.LANG_MISSING_DATA ||

                languageResult ==
                TextToSpeech.LANG_NOT_SUPPORTED

            ) {

                ttsReady =
                    false

            } else {

                ttsReady =
                    true


                tts.setSpeechRate(
                    0.90f
                )


                tts.setPitch(
                    1.0f
                )


                tts.setOnUtteranceProgressListener(

                    object :
                        UtteranceProgressListener() {


                        override fun onStart(
                            utteranceId: String?
                        ) {

                            ttsSpeaking =
                                true
                        }


                        override fun onDone(
                            utteranceId: String?
                        ) {

                            ttsSpeaking =
                                false
                        }


                        @Deprecated(
                            "Deprecated in Java"
                        )
                        override fun onError(
                            utteranceId: String?
                        ) {

                            ttsSpeaking =
                                false
                        }


                        override fun onError(
                            utteranceId: String?,
                            errorCode: Int
                        ) {

                            ttsSpeaking =
                                false
                        }
                    }
                )

            }

        } else {

            ttsReady =
                false
        }
    }



    // ============================================================
    // SIMPLE USER-FACING UI
    // ============================================================

    private fun createInterface() {

        val root =
            FrameLayout(this)


        previewView =
            PreviewView(this)


        previewView.scaleType =
            PreviewView.ScaleType.FIT_CENTER


        previewView.layoutParams =
            FrameLayout.LayoutParams(

                ViewGroup.LayoutParams.MATCH_PARENT,

                ViewGroup.LayoutParams.MATCH_PARENT
            )


        overlayView =
            DetectionOverlay(
                this
            )


        overlayView.layoutParams =
            FrameLayout.LayoutParams(

                ViewGroup.LayoutParams.MATCH_PARENT,

                ViewGroup.LayoutParams.MATCH_PARENT
            )


        statsText =
            TextView(this)


        statsText.setTextColor(
            Color.WHITE
        )


        statsText.setBackgroundColor(

            Color.argb(
                185,
                15,
                23,
                42
            )
        )


        statsText.textSize =
            17f


        statsText.text =
            "Indoor Assistant\nPoint the camera toward indoor objects. Voice guidance is automatic."


        val padding =
            (
                    16 *
                            resources
                                .displayMetrics
                                .density
                    ).toInt()


        statsText.setPadding(
            padding,
            padding,
            padding,
            padding
        )


        val params =
            FrameLayout.LayoutParams(

                ViewGroup.LayoutParams.MATCH_PARENT,

                ViewGroup.LayoutParams.WRAP_CONTENT
            )


        params.gravity =
            Gravity.TOP


        statsText.layoutParams =
            params


        root.addView(
            previewView
        )


        root.addView(
            overlayView
        )


        root.addView(
            statsText
        )


        setContentView(
            root
        )
    }


    // ============================================================
    // INITIALIZE MODEL
    // ============================================================

    private fun initializeModelAndCamera() {

        statsText.text =
            "Indoor Assistant\nLoading model..."


        cameraExecutor.execute {

            try {

                ortEnvironment =
                    OrtEnvironment.getEnvironment()


                val modelBytes =
                    assets
                        .open(MODEL_NAME)
                        .use {

                            it.readBytes()
                        }


                val options =
                    OrtSession.SessionOptions()


                options.setOptimizationLevel(

                    OrtSession
                        .SessionOptions
                        .OptLevel
                        .ALL_OPT
                )


                ortSession =
                    ortEnvironment.createSession(

                        modelBytes,

                        options
                    )


                options.close()


                val session =
                    ortSession
                        ?: throw IllegalStateException(
                            "ORT session could not be created."
                        )


                inputName =
                    session
                        .inputNames
                        .first()


                inputShortBuffer.clear()


                inputTensor =
                    OnnxTensor.createTensor(

                        ortEnvironment,

                        inputShortBuffer,

                        longArrayOf(
                            1,
                            3,
                            INPUT_SIZE.toLong(),
                            INPUT_SIZE.toLong()
                        ),

                        OnnxJavaType.FLOAT16
                    )


                runOnUiThread {

                    startCamera()
                }


            } catch (e: Exception) {

                e.printStackTrace()


                runOnUiThread {

                    statsText.text =
                        "MODEL INIT FAILED\n\n" +
                                "${e.javaClass.simpleName}\n" +
                                "${e.message}"
                }
            }
        }
    }


    // ============================================================
    // START CAMERA
    // ============================================================

    private fun startCamera() {

        val future =
            ProcessCameraProvider.getInstance(
                this
            )


        future.addListener({

            try {

                val cameraProvider =
                    future.get()


                val resolutionSelector =

                    ResolutionSelector
                        .Builder()

                        .setAspectRatioStrategy(

                            AspectRatioStrategy
                                .RATIO_4_3_FALLBACK_AUTO_STRATEGY
                        )

                        .setResolutionStrategy(

                            ResolutionStrategy(

                                Size(
                                    640,
                                    480
                                ),

                                ResolutionStrategy
                                    .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        )

                        .build()


                val preview =
                    Preview
                        .Builder()

                        .setResolutionSelector(
                            resolutionSelector
                        )

                        .build()


                preview.surfaceProvider =
                    previewView.surfaceProvider


                val analysis =
                    ImageAnalysis
                        .Builder()

                        .setResolutionSelector(
                            resolutionSelector
                        )

                        .setOutputImageFormat(

                            ImageAnalysis
                                .OUTPUT_IMAGE_FORMAT_RGBA_8888
                        )

                        .setBackpressureStrategy(

                            ImageAnalysis
                                .STRATEGY_KEEP_ONLY_LATEST
                        )

                        .build()


                analysis.setAnalyzer(

                    cameraExecutor

                ) { imageProxy ->

                    analyzeFrame(
                        imageProxy
                    )
                }


                cameraProvider.unbindAll()


                cameraProvider.bindToLifecycle(

                    this,

                    CameraSelector
                        .DEFAULT_BACK_CAMERA,

                    preview,

                    analysis
                )


                detectionHistory.clear()
                lastSpokenTime.clear()

                statsText.text =
                    "Indoor Assistant\nDetection active. Voice guidance is automatic."


            } catch (e: Exception) {

                statsText.text =
                    "CAMERA ERROR\n\n" +
                            "${e.message}"
            }

        }, ContextCompat.getMainExecutor(this))
    }



    // ============================================================
    // ANALYZE FRAME
    // ============================================================

    private fun analyzeFrame(
        imageProxy: ImageProxy
    ) {

        var cameraBitmap:
                Bitmap? =
            null


        var rotatedBitmap:
                Bitmap? =
            null


        try {

            val session =
                ortSession
                    ?: return


            val tensor =
                inputTensor
                    ?: return


            cameraBitmap =
                imageProxy.toBitmap()


            rotatedBitmap =
                rotateBitmap(

                    cameraBitmap,

                    imageProxy
                        .imageInfo
                        .rotationDegrees
                )


            val transform =
                prepareModelInput(
                    rotatedBitmap
                )


            bitmapToFp16Buffer(
                modelBitmap
            )


            val result =
                session.run(

                    mapOf(
                        inputName to
                                tensor
                    )
                )


            val outputTensor =
                result[0] as OnnxTensor


            val detections =
                fastDecodeAndNms(

                    outputTensor,

                    transform
                )


            result.close()


            processTemporalConfirmation(
                detections
            )


            runOnUiThread {

                overlayView.updateDetections(

                    detections,

                    transform.originalWidth,

                    transform.originalHeight
                )
            }


        } catch (e: Exception) {

            e.printStackTrace()


        } finally {

            imageProxy.close()


            if (

                rotatedBitmap != null &&
                rotatedBitmap !== cameraBitmap

            ) {

                rotatedBitmap.recycle()
            }


            cameraBitmap?.recycle()
        }
    }


    // ============================================================
    // TEMPORAL CONFIRMATION
    // ============================================================

    private fun processTemporalConfirmation(
        detections: List<Detection>
    ) {

        /*
         * Store only the class IDs seen in the current frame.
         * This keeps the existing 2-of-3 temporal confirmation logic.
         */
        val currentClasses =
            detections
                .map { it.classId }
                .toSet()


        detectionHistory.addLast(
            currentClasses
        )


        while (
            detectionHistory.size >
            HISTORY_SIZE
        ) {

            detectionHistory.removeFirst()
        }


        if (
            detectionHistory.size <
            HISTORY_SIZE
        ) {

            return
        }


        val counts =
            IntArray(
                NUM_CLASSES
            )


        for (
            frameClasses in
            detectionHistory
        ) {

            for (
                classId in
                frameClasses
            ) {

                counts[
                    classId
                ] += 1
            }
        }


        val confirmedClassIds =
            mutableSetOf<Int>()


        for (
            classId in
            0 until NUM_CLASSES
        ) {

            if (
                counts[classId] >=
                REQUIRED_CONFIRMATIONS
            ) {

                confirmedClassIds.add(
                    classId
                )
            }
        }


        if (
            confirmedClassIds.isEmpty()
        ) {

            return
        }


        /*
         * Only announce objects that are BOTH:
         * 1) confirmed in at least 2 of the last 3 frames, and
         * 2) still visible in the current frame.
         *
         * If there are multiple boxes of the same class, keep the
         * highest-confidence one for speech so we do not say the
         * same class repeatedly (for example, "chair, chair, chair").
         */
        val currentConfirmedByClass =
            detections
                .filter {
                    confirmedClassIds.contains(
                        it.classId
                    )
                }
                .groupBy {
                    it.classId
                }
                .mapNotNull { (_, classDetections) ->
                    classDetections.maxByOrNull {
                        it.confidence
                    }
                }


        if (
            currentConfirmedByClass.isEmpty()
        ) {

            return
        }


        attemptVoiceWarnings(
            currentConfirmedByClass
        )
    }


    // ============================================================
    // VOICE WARNING
    // ============================================================

    private fun attemptVoiceWarnings(
        detections: List<Detection>
    ) {

        if (
            !ttsReady ||
            ttsSpeaking
        ) {

            return
        }


        val now =
            System.currentTimeMillis()


        /*
         * Respect the existing 5-second cooldown PER CLASS.
         * A class that was just spoken is skipped, while other newly
         * confirmed objects can still be announced in the same frame.
         */
        val eligible =
            detections.filter { detection ->

                val last =
                    lastSpokenTime[
                        detection.classId
                    ] ?: 0L


                now - last >=
                    SPEECH_COOLDOWN_MS
            }


        if (
            eligible.isEmpty()
        ) {

            return
        }


        /*
         * Safety objects are placed first, then ordinary objects.
         * Within each group, higher-confidence detections are spoken first.
         */
        val ordered =
            eligible.sortedWith(
                compareByDescending<Detection> {
                    SAFETY_CLASS_IDS.contains(
                        it.classId
                    )
                }
                    .thenByDescending {
                        it.confidence
                    }
            )


        val spokenSuccessfully =
            speakDetections(
                ordered
            )


        if (
            spokenSuccessfully
        ) {

            for (
                detection in
                ordered
            ) {

                lastSpokenTime[
                    detection.classId
                ] = now
            }
        }
    }


    private fun speakDetections(
        detections: List<Detection>
    ): Boolean {

        val tts =
            textToSpeech
                ?: return false


        if (
            detections.isEmpty()
        ) {

            return false
        }


        val safetyNames =
            detections
                .filter {
                    SAFETY_CLASS_IDS.contains(
                        it.classId
                    )
                }
                .map {
                    classNames[
                        it.classId
                    ]
                }
                .distinct()


        val ordinaryNames =
            detections
                .filterNot {
                    SAFETY_CLASS_IDS.contains(
                        it.classId
                    )
                }
                .map {
                    classNames[
                        it.classId
                    ]
                }
                .distinct()


        val messageParts =
            mutableListOf<String>()


        if (
            safetyNames.isNotEmpty()
        ) {

            messageParts.add(
                "Warning. ${formatObjectList(safetyNames)} ahead."
            )
        }


        if (
            ordinaryNames.isNotEmpty()
        ) {

            messageParts.add(
                "${formatObjectList(ordinaryNames)} ahead."
            )
        }


        val message =
            messageParts.joinToString(
                separator = " "
            )


        if (
            message.isBlank()
        ) {

            return false
        }


        speechCounter += 1

        ttsSpeaking = true


        val result =
            tts.speak(
                message,
                TextToSpeech.QUEUE_FLUSH,
                null,
                "indoor_assistant_$speechCounter"
            )


        if (
            result ==
            TextToSpeech.ERROR
        ) {

            ttsSpeaking = false
            return false
        }


        return true
    }


    private fun formatObjectList(
        names: List<String>
    ): String {

        return when (
            names.size
        ) {

            0 ->
                ""

            1 ->
                names[0]

            2 ->
                "${names[0]} and ${names[1]}"

            else ->
                names
                    .dropLast(1)
                    .joinToString(
                        separator = ", "
                    ) +
                    ", and ${names.last()}"
        }
    }


    // ============================================================
    // ROTATION
    // ============================================================

    private fun rotateBitmap(

        source: Bitmap,

        degrees: Int

    ): Bitmap {

        if (
            degrees == 0
        ) {

            return source
        }


        val matrix =
            Matrix()


        matrix.postRotate(
            degrees.toFloat()
        )


        return Bitmap.createBitmap(

            source,

            0,

            0,

            source.width,

            source.height,

            matrix,

            true
        )
    }


    // ============================================================
    // LETTERBOX
    // ============================================================

    private fun prepareModelInput(
        source: Bitmap
    ): FrameTransform {

        val width =
            source.width


        val height =
            source.height


        val scale =
            min(

                INPUT_SIZE.toFloat() /
                        width,

                INPUT_SIZE.toFloat() /
                        height
            )


        val resizedWidth =
            width *
                    scale


        val resizedHeight =
            height *
                    scale


        val padX =
            (
                    INPUT_SIZE -
                            resizedWidth
                    ) / 2f


        val padY =
            (
                    INPUT_SIZE -
                            resizedHeight
                    ) / 2f


        modelCanvas.drawColor(

            Color.rgb(
                114,
                114,
                114
            )
        )


        val destination =
            RectF(

                padX,

                padY,

                padX +
                        resizedWidth,

                padY +
                        resizedHeight
            )


        modelCanvas.drawBitmap(

            source,

            null,

            destination,

            resizePaint
        )


        return FrameTransform(

            originalWidth =
                width,

            originalHeight =
                height,

            scale =
                scale,

            padX =
                padX,

            padY =
                padY
        )
    }


    // ============================================================
    // FP16 PREPROCESSING
    // ============================================================

    private fun bitmapToFp16Buffer(
        bitmap: Bitmap
    ) {

        bitmap.getPixels(

            pixelBuffer,

            0,

            INPUT_SIZE,

            0,

            0,

            INPUT_SIZE,

            INPUT_SIZE
        )


        inputShortBuffer.clear()


        val planeSize =
            INPUT_SIZE *
                    INPUT_SIZE


        for (
        i in pixelBuffer.indices
        ) {

            val pixel =
                pixelBuffer[i]


            val red =
                (
                        pixel shr 16
                        ) and 0xFF


            val green =
                (
                        pixel shr 8
                        ) and 0xFF


            val blue =
                pixel and 0xFF


            inputShortBuffer.put(

                i,

                fp16Lookup[
                    red
                ]
            )


            inputShortBuffer.put(

                planeSize + i,

                fp16Lookup[
                    green
                ]
            )


            inputShortBuffer.put(

                2 *
                        planeSize +
                        i,

                fp16Lookup[
                    blue
                ]
            )
        }


        inputShortBuffer.position(
            0
        )
    }


    // ============================================================
    // YOLO DECODER + NMS
    // ============================================================

    private fun fastDecodeAndNms(

        outputTensor: OnnxTensor,

        transform: FrameTransform

    ): List<Detection> {

        val floatBuffer =
            outputTensor.floatBuffer
                ?: throw IllegalStateException(
                    "Unable to read model output."
                )


        floatBuffer.rewind()


        floatBuffer.get(
            rawOutput
        )


        val candidatesByClass =
            Array(
                NUM_CLASSES
            ) {

                ArrayList<Detection>()
            }


        for (
        index in
        0 until NUM_PREDICTIONS
        ) {

            var bestClass =
                -1


            var bestConfidence =
                0f


            for (
            classId in
            0 until NUM_CLASSES
            ) {

                val confidence =

                    rawOutput[

                        (
                                4 +
                                        classId
                                ) *
                                NUM_PREDICTIONS +
                                index
                    ]


                if (
                    confidence >
                    bestConfidence
                ) {

                    bestConfidence =
                        confidence

                    bestClass =
                        classId
                }
            }


            if (
                bestClass < 0
            ) {

                continue
            }


            if (

                bestConfidence <
                thresholdForClass(
                    bestClass
                )

            ) {

                continue
            }


            val cx =
                rawOutput[
                    index
                ]


            val cy =
                rawOutput[
                    NUM_PREDICTIONS +
                            index
                ]


            val w =
                rawOutput[
                    2 *
                            NUM_PREDICTIONS +
                            index
                ]


            val h =
                rawOutput[
                    3 *
                            NUM_PREDICTIONS +
                            index
                ]


            var x1 =
                cx -
                        w / 2f


            var y1 =
                cy -
                        h / 2f


            var x2 =
                cx +
                        w / 2f


            var y2 =
                cy +
                        h / 2f


            x1 =
                (
                        x1 -
                                transform.padX
                        ) /
                        transform.scale


            y1 =
                (
                        y1 -
                                transform.padY
                        ) /
                        transform.scale


            x2 =
                (
                        x2 -
                                transform.padX
                        ) /
                        transform.scale


            y2 =
                (
                        y2 -
                                transform.padY
                        ) /
                        transform.scale


            x1 =
                x1.coerceIn(
                    0f,
                    transform.originalWidth.toFloat()
                )


            y1 =
                y1.coerceIn(
                    0f,
                    transform.originalHeight.toFloat()
                )


            x2 =
                x2.coerceIn(
                    0f,
                    transform.originalWidth.toFloat()
                )


            y2 =
                y2.coerceIn(
                    0f,
                    transform.originalHeight.toFloat()
                )


            if (
                x2 <= x1 ||
                y2 <= y1
            ) {

                continue
            }


            candidatesByClass[
                bestClass
            ].add(

                Detection(

                    classId =
                        bestClass,

                    confidence =
                        bestConfidence,

                    x1 =
                        x1,

                    y1 =
                        y1,

                    x2 =
                        x2,

                    y2 =
                        y2
                )
            )
        }


        val finalDetections =
            ArrayList<Detection>()


        for (
        classId in
        0 until NUM_CLASSES
        ) {

            val candidates =
                candidatesByClass[
                    classId
                ]


            if (
                candidates.isEmpty()
            ) {

                continue
            }


            candidates.sortByDescending {
                it.confidence
            }


            val suppressed =
                BooleanArray(
                    candidates.size
                )


            for (
            i in
            candidates.indices
            ) {

                if (
                    suppressed[
                        i
                    ]
                ) {

                    continue
                }


                val current =
                    candidates[
                        i
                    ]


                finalDetections.add(
                    current
                )


                if (
                    finalDetections.size >=
                    MAX_DETECTIONS
                ) {

                    break
                }


                for (
                j in
                i + 1 until
                        candidates.size
                ) {

                    if (
                        suppressed[
                            j
                        ]
                    ) {

                        continue
                    }


                    if (

                        calculateIoU(

                            current,

                            candidates[
                                j
                            ]

                        ) >
                        NMS_IOU_THRESHOLD

                    ) {

                        suppressed[
                            j
                        ] =
                            true
                    }
                }
            }
        }


        finalDetections.sortByDescending {
            it.confidence
        }


        return finalDetections
    }


    // ============================================================
    // IOU
    // ============================================================

    private fun calculateIoU(

        a: Detection,

        b: Detection

    ): Float {

        val x1 =
            max(
                a.x1,
                b.x1
            )


        val y1 =
            max(
                a.y1,
                b.y1
            )


        val x2 =
            min(
                a.x2,
                b.x2
            )


        val y2 =
            min(
                a.y2,
                b.y2
            )


        val width =
            x2 -
                    x1


        val height =
            y2 -
                    y1


        if (
            width <= 0f ||
            height <= 0f
        ) {

            return 0f
        }


        val intersection =
            width *
                    height


        val areaA =
            (
                    a.x2 -
                            a.x1
                    ) *
                    (
                            a.y2 -
                                    a.y1
                            )


        val areaB =
            (
                    b.x2 -
                            b.x1
                    ) *
                    (
                            b.y2 -
                                    b.y1
                            )


        val union =
            areaA +
                    areaB -
                    intersection


        if (
            union <= 0f
        ) {

            return 0f
        }


        return intersection /
                union
    }


    // ============================================================
    // DETECTION OVERLAY
    // ============================================================

    inner class DetectionOverlay(
        context: Context
    ) : View(context) {

        private var detections:
                List<Detection> =
            emptyList()


        private var frameWidth =
            1


        private var frameHeight =
            1


        private val boxPaint =
            Paint().apply {

                color =
                    Color.RED

                style =
                    Paint.Style.STROKE

                strokeWidth =
                    5f
            }


        private val textPaint =
            Paint().apply {

                color =
                    Color.WHITE

                textSize =
                    34f
            }


        private val backgroundPaint =
            Paint().apply {

                color =
                    Color.argb(
                        210,
                        0,
                        0,
                        0
                    )
            }


        fun updateDetections(

            values:
            List<Detection>,

            width: Int,

            height: Int

        ) {

            detections =
                values


            frameWidth =
                max(
                    1,
                    width
                )


            frameHeight =
                max(
                    1,
                    height
                )


            invalidate()
        }


        override fun onDraw(
            canvas: Canvas
        ) {

            super.onDraw(
                canvas
            )


            val scale =
                min(

                    width.toFloat() /
                            frameWidth,

                    height.toFloat() /
                            frameHeight
                )


            val displayWidth =
                frameWidth *
                        scale


            val displayHeight =
                frameHeight *
                        scale


            val offsetX =
                (
                        width -
                                displayWidth
                        ) / 2f


            val offsetY =
                (
                        height -
                                displayHeight
                        ) / 2f


            for (
            detection in
            detections
            ) {

                val left =
                    offsetX +
                            detection.x1 *
                            scale


                val top =
                    offsetY +
                            detection.y1 *
                            scale


                val right =
                    offsetX +
                            detection.x2 *
                            scale


                val bottom =
                    offsetY +
                            detection.y2 *
                            scale


                canvas.drawRect(

                    left,

                    top,

                    right,

                    bottom,

                    boxPaint
                )


                val label =

                    classNames[
                        detection.classId
                    ] +

                            " " +

                            String.format(
                                Locale.US,
                                "%.2f",
                                detection.confidence
                            )


                val textWidth =
                    textPaint.measureText(
                        label
                    )


                val labelTop =
                    max(
                        0f,
                        top -
                                42f
                    )


                canvas.drawRect(

                    left,

                    labelTop,

                    left +
                            textWidth +
                            15f,

                    labelTop +
                            42f,

                    backgroundPaint
                )


                canvas.drawText(

                    label,

                    left +
                            6f,

                    labelTop +
                            32f,

                    textPaint
                )
            }
        }
    }


    // ============================================================
    // CLEANUP
    // ============================================================

    override fun onDestroy() {

        try {

            textToSpeech?.stop()

            textToSpeech?.shutdown()

        } catch (_: Exception) {

        }


        try {

            inputTensor?.close()

        } catch (_: Exception) {

        }


        try {

            ortSession?.close()

        } catch (_: Exception) {

        }


        cameraExecutor.shutdown()


        super.onDestroy()
    }
}