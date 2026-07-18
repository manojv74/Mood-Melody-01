# Mood Melody

### Emotion-aware music generation for Android

Mood Melody is an end-to-end AI application that detects a user's facial expression on an Android device and generates a short music clip suited to the detected mood. It combines on-device computer vision, a Flask REST API, and Meta's MusicGen model in one mobile-to-backend workflow.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Kotlin](https://img.shields.io/badge/Kotlin-Android-7F52FF?logo=kotlin&logoColor=white)](https://kotlinlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The application uses a MobileNetV3-based classifier trained on FER+ to recognize eight facial-expression classes. The trained model is converted for on-device inference, allowing the Android application to process camera frames locally. The detected mood and confidence are then sent to a Flask backend, which turns them into a music prompt and generates a 10-second WAV clip with `facebook/musicgen-small`.

### Supported expressions

`Neutral` · `Happiness` · `Surprise` · `Sadness` · `Anger` · `Disgust` · `Fear` · `Contempt`

## Key features

- Continuous facial-expression monitoring with CameraX
- On-device inference using a mobile-optimized MobileNetV3 model
- Face detection and tracking with Google ML Kit
- Confidence-based aggregation to stabilize frame-by-frame predictions
- Mood-conditioned music generation using MusicGen
- Flask REST API connecting the Android client to the generation service
- In-app playback of generated WAV audio, with local-audio selection support
- Optional Gemini-assisted prompt enhancement with a built-in fallback

## System workflow

```mermaid
flowchart LR
    A[CameraX input] --> B[ML Kit face detection]
    B --> C[On-device emotion model]
    C --> D[Emotion aggregation]
    D --> E[Flask API]
    E --> F[MusicGen]
    F --> G[Audio playback]
```

1. The Android app captures and analyzes frames from the front camera.
2. ML Kit locates the face, and the mobile model classifies the expression.
3. Predictions are aggregated to reduce rapid changes between frames.
4. The app sends the selected mood and confidence to `POST /generate`.
5. The backend builds a mood-aware prompt and generates a WAV file.
6. The generated file URL is returned to the app for playback.

## Tech stack

| Layer | Technologies |
| --- | --- |
| Android application | Kotlin, Jetpack Compose, CameraX, Google ML Kit, OkHttp |
| Mobile inference | LiteRT / TensorFlow Lite |
| Model development | Python, PyTorch, torchvision, MobileNetV3, OpenCV |
| Backend | Flask, Flask-CORS |
| Music generation | Hugging Face Transformers, `facebook/musicgen-small` |
| Audio processing | NumPy, SciPy |

## Repository structure

```text
Mood-Melody-01/
├── android_app/          # Kotlin Android application
├── backend/              # Flask API and MusicGen integration
├── emotion-detection/    # Training, evaluation, and model-conversion work
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- Git
- Python 3.10 or later
- Android Studio with JDK 17
- Android device or emulator running API 28 or later
- A CUDA-capable GPU is recommended for practical MusicGen inference

### 1. Clone the repository

```bash
git clone https://github.com/manojv74/Mood-Melody-01.git
cd Mood-Melody-01
```

### 2. Run the backend

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the backend dependencies and start the server:

```bash
pip install -r backend/requirements.txt
python backend/music_server.py
```

The server runs at `http://localhost:5000` by default. The first launch downloads the MusicGen weights and may take additional time.

Optional environment variables:

```text
GEMINI_API_KEY=your_api_key   # Enables prompt enhancement
PORT=5000                     # Overrides the default port
```

The Gemini key is optional; the backend falls back to its internal prompt builder when the key is unavailable.

### 3. Run the Android application

1. Open the `android_app/` directory in Android Studio.
2. Allow Gradle to sync and download the required dependencies.
3. In `MainActivity.kt`, set `SERVER_BASE_URL` for your environment:
   - Android emulator: `http://10.0.2.2:5000`
   - Physical device: use the development computer's LAN address
   - Production: use a public HTTPS backend URL
4. Build and run the application on an emulator or device.

You can also install the debug build from a terminal:

```bash
cd android_app
./gradlew installDebug
```

On Windows, use `gradlew.bat installDebug`.

## API reference

### Generate music

`POST /generate`

Example request:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"default","mood":"happiness","confidence":0.92}'
```

Example response:

```json
{
  "prompt": "upbeat pop, joyful",
  "music_url": "http://localhost:5000/music/generated_track.wav"
}
```

## Model development

The emotion classifier uses a pretrained MobileNetV3-Large backbone with a custom eight-class classification head. Training is configured for 224 × 224 inputs and FER+ labels.

To retrain the model:

```bash
cd emotion-detection
pip install -r requirements.txt
python train_model.py
```

Before running the script locally, update the dataset and output paths in `train_model.py`; the current defaults use Kaggle directories.

> Model accuracy is intentionally not stated here because a reproducible evaluation result should be reported only alongside the exact dataset split, preprocessing pipeline, checkpoint, and evaluation script.

## Current limitations

- MusicGen is compute- and memory-intensive, especially without CUDA.
- The backend stores generated tracks locally and requires a cleanup or object-storage strategy for production use.
- The Android server URL is currently configured in source code.
- Local HTTP is suitable for development; a deployed Android client should communicate over HTTPS.

## Future improvements

- Add reproducible model evaluation and publish a classification report and confusion matrix
- Move server configuration into Android build settings
- Add backend tests and continuous integration
- Introduce a lightweight mock generator for development and automated testing
- Store generated audio in object storage with automatic expiration
- Add user-selectable musical styles and generation controls

## Author

**Manoj V**

- GitHub: [@manojv74](https://github.com/manojv74)
- LinkedIn: [linkedin.com/in/manoj-v74](https://www.linkedin.com/in/manoj-v74/)

## License

This project is licensed under the [MIT License](LICENSE).
