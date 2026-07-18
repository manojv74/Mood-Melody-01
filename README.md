# Mood Melody

Mood Melody is an emotion-based music generation system. The Android app detects a user's facial emotion on-device and the Python backend generates a short music clip that matches the detected mood. This README is beginner-friendly and explains what the project is, how it works, and how to run each part locally.

---

## Project overview

Mood Melody captures a face image on an Android device, classifies the facial emotion using a lightweight model (MobileNetV3 converted to TensorFlow Lite), sends the detected emotion to a Flask backend, and the backend generates a short WAV music clip using MusicGen that matches the emotion. The Android app downloads and plays the generated audio.

---

## Architecture / workflow (simple)

1. Android app captures a face image (CameraX) and runs the TFLite emotion classifier → outputs `mood` + `confidence`.
2. Android POSTs JSON to the backend `/generate` endpoint with `{ user_id, mood, confidence }`.
3. Backend composes a prompt (optionally enhanced by an LLM), runs MusicGen to synthesize audio, saves a WAV under `static/music/`, and returns the file URL.
4. Android downloads and plays the returned audio.

---

## Project structure

- `backend/` — Flask backend server and MusicGen integration (API endpoints).
- `emotion-detection/` — model training, evaluation, and conversion scripts (PyTorch → ONNX/TF → TFLite).
- `android_app/` — Android mobile application (Kotlin, Gradle, CameraX, TFLite integration).
- `README.md`, `LICENSE`, `.gitignore` — top-level repository files.

---

## Main features

- Real-time facial emotion detection on Android
- Emotion classification using a MobileNetV3-based model
- Music generation using MusicGen based on detected emotion
- Flask REST API for generation requests
- Android interface for capture and playback
- Audio file generation (WAV) and playback in-app

---

## Tech stack

- Python, Flask, Flask-CORS
- PyTorch (training), torchvision
- TensorFlow Lite (mobile inference)
- OpenCV (image preprocessing)
- MusicGen / transformers (audio generation)
- NumPy, SciPy
- Android, Kotlin, Gradle

---

## Important project details

- Emotion detector: MobileNetV3-based classifier (trained in PyTorch).
- Training data: FER2013 / FER2013+ (emotion classes include happy, sad, angry, neutral, fear, surprise, disgust, contempt).
- Model conversion: trained PyTorch model is converted to TFLite for Android integration.
- Backend: receives emotion input and uses MusicGen to produce short music clips.

---

## Setup — Backend

1. Clone the repository and open a terminal:

```bash
git clone https://github.com/manojv74/Mood-Melody-01.git
cd Mood-Melody-
```

2. Create and activate a Python virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

4. Run the backend server (example):

```bash
python backend/music_server.py
```

- Default port: `5000`. You can change it with the `PORT` environment variable.
- Note: MusicGen model weights are large and GPU-intensive. For local development, consider mocking generation or using a smaller model.

---

## Setup — Emotion detection (training & conversion)

1. Change into the folder:

```bash
cd emotion-detection
```

2. Create/activate a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Training and validation:

- Review `train_model.py` for dataset path settings (`DATA_DIR`, CSV). The script was developed with Kaggle-style paths — update those constants or pass dataset paths before running locally.
- Run the training script after preparing FER2013/FER2013+ data:

```bash
python train_model.py
```

4. Convert trained model to TFLite (high-level steps):

- Export the trained PyTorch model to ONNX or TorchScript.
- Convert ONNX → TensorFlow SavedModel (if using ONNX → TF), then SavedModel → TFLite using `tf.lite.TFLiteConverter`.
- Optionally apply quantization to reduce size and improve performance on-device.

If you want, I can add a conversion helper script that follows your exact training outputs.

---

## Setup — Android app (Android Studio)

1. Open Android Studio and select `File → Open` then choose the `android_app/` directory.
2. Allow Gradle to sync and download dependencies.
3. Configure the backend base URL in the app's network/Retrofit settings.
   - For emulator -> host machine: use `http://10.0.2.2:5000`.
4. Build and run on an emulator or device via Android Studio.
   - Or from terminal:

```bash
cd android_app
./gradlew installDebug
```

Note: Android dependencies are managed by Gradle — there is no `requirements.txt` for the Android app.

---

## API flow (end-to-end)

1. Android app detects emotion: `{ "mood": "happiness", "confidence": 0.92 }`.
2. Android sends POST request to backend `/generate` with `user_id`, `mood`, `confidence`.
3. Backend builds a prompt and runs MusicGen to create a WAV file (saved in `static/music/`).
4. Backend responds with `prompt` and `music_url` JSON fields.
5. Android downloads the WAV from `music_url` and plays it.

---

## Sample API request & response

Request (curl):

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"default","mood":"happiness","confidence":0.92}'
```

Response (example):

```json
{
  "prompt": "bright acoustic guitar, upbeat tempo — happy mood",
  "music_url": "http://localhost:5000/music/music_gen_2026-07-09T12-34-56.wav"
}
```

---

## Install requirements (commands)

- Backend:

```bash
pip install -r backend/requirements.txt
```

- Emotion detection (training):

```bash
cd emotion-detection
pip install -r requirements.txt
```

- Android: dependencies handled by Gradle (no `requirements.txt`).

---

## Future improvements (suggested)

- Provide a small pre-built TFLite model for quick Android testing.
- Add a mock or lightweight generator option for CI / development (no GPU required).
- Make `train_model.py` accept CLI arguments or environment variables instead of hardcoded Kaggle paths.
- Move generated audio to object storage (e.g., S3) for better serving and scaling.
- Add unit tests and a simple CI pipeline to run backend smoke tests and linting.

---

## License

This project is released under the MIT License — see the `LICENSE` file in the repository root for details.

