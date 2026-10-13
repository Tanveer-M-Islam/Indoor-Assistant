# Indoor Assistant

Indoor Assistant is an Android-based assistive object detection application
designed to support visually impaired users in indoor environments.

The application uses a lightweight YOLO11n model with on-device ONNX Runtime
inference and automatic Text-to-Speech guidance.

## Download APK

Download the latest Android application from the
[GitHub Releases](../../releases/latest) page.

> Note: This application is distributed outside Google Play.
> Android or Google Play Protect may show a warning when installing the APK.

## Features

- Real-time indoor object detection
- YOLO11n object detector
- FP16 ONNX deployment
- 16 indoor and safety-related object classes
- Automatic Text-to-Speech feedback
- Multiple detected object classes announced together
- Safety-critical object prioritization
- Class-specific confidence thresholds
- 2-of-3 temporal confirmation
- 5-second per-class speech cooldown
- On-device inference

## Object Classes

- Bed
- Sofa
- Chair
- Table
- Lamp
- TV
- Laptop
- Wardrobe
- Window
- Door
- Potted Plant
- Photo Frame
- Scissors
- Knife
- Cutter
- Syringe

## Safety-Critical Objects

The following classes receive priority during voice feedback:

- Scissors
- Knife
- Cutter
- Syringe

## Model

- Model: YOLO11n
- Deployment format: FP16 ONNX
- Input resolution: 640 × 640
- Model size: approximately 5.11 MB

## Android Application

The Android application performs:

CameraX → Image preprocessing → YOLO inference → NMS →
Temporal confirmation → Safety prioritization → Text-to-Speech

## Installation

1. Open the **Releases** section of this repository.
2. Download `IndoorAssistant-v1.0.0.apk`.
3. Open the APK on an Android phone.
4. Allow installation from unknown sources if requested.
5. Android/Play Protect may show a warning because the application is sideloaded.
6. Grant camera permission.
7. Open Indoor Assistant.

## Research Project

This application was developed as part of an undergraduate thesis on
lightweight and explainable indoor object detection for assistive systems.

## Disclaimer

Indoor Assistant is a research prototype.

It is not a certified medical, navigation, or personal-safety device.