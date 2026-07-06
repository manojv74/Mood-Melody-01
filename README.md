# 🎵 Mood Melody

Mood Melody is an AI-powered music generation project that creates music based on a user's facial emotion.

The idea behind this project is simple: instead of asking users to manually select their mood, the application detects their facial expression, predicts the emotion, and generates music that matches it.

This project combines computer vision, deep learning, Android development, backend development, and generative AI into a single workflow.

---

## Project Overview

The complete system consists of three major modules:

* **Emotion Detection Model** – Detects facial emotion from an image.
* **Android Application** – Captures the user's face and communicates with the backend.
* **Backend Server** – Generates music based on the detected emotion using AI.

---

## How It Works

```text
User
   │
   ▼
Android App
   │
   ▼
Capture Face
   │
   ▼
TensorFlow Lite Emotion Model
   │
   ▼
Emotion + Confidence Score
   │
   ▼
POST Request (REST API)
   │
   ▼
Flask Backend
   │
   ▼
Generate Music Prompt
   │
   ▼
Gemini API
   │
   ▼
Enhanced Prompt
   │
   ▼
MusicGen
   │
   ▼
Generate WAV File
   │
   ▼
Return Music URL
   │
   ▼
Android App
   │
   ▼
Play Music
```

---

## Project Structure

```text
Mood-Melody/
│
├── android_app/
│   └── Android application
│
├── backend/
│   └── Flask server and MusicGen integration
│
├── emotion-detection/
│   └── Emotion detection model training
│
└── README.md
```

---

## Technologies Used

### Machine Learning

* Python
* PyTorch
* MobileNetV3Large
* TensorFlow Lite

### Backend

* Flask
* REST API
* MusicGen
* Gemini API

### Android

* Kotlin
* CameraX
* TensorFlow Lite
* Retrofit

---

## Emotion Detection

The emotion detection model is trained using the FER+ dataset. During training, images are preprocessed, augmented, and passed through a MobileNetV3Large model using transfer learning.

The trained model is converted to TensorFlow Lite so that it can run efficiently on an Android device.

The model predicts one of the following emotions:

* Neutral
* Happiness
* Surprise
* Sadness
* Anger
* Disgust
* Fear
* Contempt

---

## Backend

The backend is built using Flask.

After receiving the detected emotion and confidence score from the Android application, it creates a music prompt, enhances it using the Gemini API, and passes the prompt to MusicGen to generate a short audio clip.

The generated audio is stored as a WAV file, and the backend returns its URL to the Android application.

---

## Android Application

The Android application captures the user's face using the device camera.

The TensorFlow Lite model runs locally on the device to predict the user's emotion. The prediction is then sent to the Flask backend, which returns the generated music. The application downloads the audio and plays it automatically.

---

## Future Improvements

* Improve emotion detection accuracy using a larger dataset.
* Support longer music generation.
* Personalize recommendations based on user listening history.
* Deploy the backend on a cloud platform for public access.

---

## Note

This project was developed as a learning project to explore the integration of deep learning, backend development, Android development, and generative AI in a single end-to-end application.
