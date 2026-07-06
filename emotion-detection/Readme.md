# Emotion Detection Model

## Overview

This module is responsible for training the facial emotion recognition model used in the Mood Melody project.

The model is trained on the FER+ dataset using transfer learning with MobileNetV3Large. It predicts one of eight facial emotions from a face image, and the predicted emotion along with its confidence score is later used by the Android application to generate personalized music through the backend.

---

## Features

* Transfer Learning using MobileNetV3Large
* FER+ facial emotion dataset
* Data augmentation for better generalization
* Class balancing using weighted CrossEntropy Loss
* Two-stage fine-tuning strategy
* Validation-based model checkpointing
* Early stopping to reduce overfitting
* Performance evaluation using Accuracy, Classification Report, and Confusion Matrix

---

## Emotion Classes

The model predicts the following emotions:

* Neutral
* Happiness
* Surprise
* Sadness
* Anger
* Disgust
* Fear
* Contempt

---

## Model Pipeline

```text
FER+ Dataset
      │
      ▼
Load Images & Labels
      │
      ▼
Image Preprocessing
(Resize → RGB Conversion → Normalization)
      │
      ▼
Data Augmentation
      │
      ▼
MobileNetV3Large
      │
      ▼
Training
      │
      ▼
Validation
      │
      ▼
Best Model Saved
      │
      ▼
Evaluation on Test Set
```

---

## Image Preprocessing

Each input image goes through the following preprocessing steps before being passed to the model:

* Resize images to **224 × 224**
* Convert grayscale images to RGB
* Convert images into tensors
* Normalize pixel values
* Apply augmentation during training

Training augmentations include:

* Random Crop
* Horizontal Flip
* Rotation
* Translation
* Brightness Adjustment
* Contrast Adjustment

---

## Model Architecture

Backbone:

* MobileNetV3Large (Pre-trained)

Custom Classification Head:

```text
MobileNetV3 Features
        │
Linear (512)
        │
ReLU
        │
Dropout
        │
Linear (256)
        │
ReLU
        │
Dropout
        │
Linear (8 Emotion Classes)
```

---

## Training Strategy

The model is trained in two phases.

### Phase 1

* Freeze MobileNetV3 feature extractor
* Train only the custom classifier

### Phase 2

* Unfreeze the complete network
* Fine-tune the entire model using a lower learning rate

---

## Training Configuration

| Parameter      |                     Value |
| -------------- | ------------------------: |
| Model          |          MobileNetV3Large |
| Dataset        |                      FER+ |
| Image Size     |                 224 × 224 |
| Batch Size     |                        32 |
| Epochs         |                        25 |
| Optimizer      |                      Adam |
| Loss Function  | Weighted CrossEntropyLoss |
| Learning Rate  |            0.001 → 0.0001 |
| Scheduler      |                    StepLR |
| Early Stopping |                       Yes |

---

## Evaluation Metrics

The trained model is evaluated using:

* Test Accuracy
* Classification Report
* Confusion Matrix
* Training Loss
* Validation Loss

---

## Project Role

This module is responsible only for emotion recognition.

The trained model predicts:

* Emotion Label
* Confidence Score

These predictions are later used by the backend to generate personalized music.

```text
Face Image
      │
      ▼
Emotion Detection Model
      │
      ▼
Emotion + Confidence
   

