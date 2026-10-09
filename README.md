# Offline Gesture Recognition

A cross-platform desktop application that recognizes ten predefined hand gestures in real time from a local camera. Built with Python, PySide6, and OpenCV. Runs entirely offline: frames are processed in memory and nothing is stored or transmitted.

## Recognized gestures

- Swipe Left / Swipe Right
- Peace Sign
- Close Hand
- Open Hand
- Wave Left / Wave Right with Index Finger Raised
- Wave Left / Wave Right with Index and Middle Fingers Raised
- Join Both Hands in a Clap/Prayer Gesture

## Requirements

- Python 3.10 to 3.12
- A webcam
- Windows, macOS, or Linux

## Setup

1. Create and activate a virtual environment from the project root:

```
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate           # Windows
```

2. Install dependencies:

```
   pip install -r requirements.txt
```

3. Download the hand landmark model into `gesture_app/models/`. This is a one-time step, and the application never downloads anything at runtime:

```
   mkdir -p gesture_app/models
   curl -L -o gesture_app/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

   On Windows, download the same URL in a browser and save it as `gesture_app\models\hand_landmarker.task`.

## Running

From the project root, with the virtual environment active:

```
python -m gesture_app.main
```

Click **Start** to begin recognition and **Stop** to end it. The five indicators show the last gesture, the current gesture, confidence, camera FPS, and processing FPS.

## Configuration

Settings are read from `~/.gesture_app/config.json` if the file exists. Any missing or invalid value falls back to its default. Useful keys include:

- `camera_index`: which camera to use (default `0`)
- `confidence_threshold`: minimum confidence to accept a gesture (default `0.7`)
- `swipe_min_dx`, `wave_min_dx`, `join_dist`: gesture sensitivity thresholds, which usually need tuning for your camera and setup

Diagnostic logs are written to `~/.gesture_app/logs/app.log`. They never contain camera frames or landmark data.

## Privacy

Camera frames and recognition data stay in memory. The application makes no network calls at runtime, collects no telemetry, and writes no images to disk.

## Project structure

```
gesture_app/
  domain.py          shared data contracts and gesture identifiers
  config.py          configuration loading and validation
  logging_setup.py   non-blocking diagnostic logging
  camera.py          camera capture and latest-frame buffer
  vision.py          hand detection and landmarks
  tracking.py        hand tracking and motion history
  recognition.py     static, temporal, and two-hand recognizers
  decision.py        confidence checks, duplicate suppression, events
  processing.py      processing worker thread
  controller.py      application lifecycle and worker coordination
  ui.py              PySide6 interface
  main.py            entry point
  models/            local model files
requirements.txt
```

## Troubleshooting

- **"Hand landmark model not found"**: complete the model download step in Setup.
- **"Camera unavailable"**: check that no other application is using the camera, or change `camera_index` in the config file.
- **Gestures trigger too easily or not at all**: adjust the thresholds in the config file.

## Status

Version 1. Gesture thresholds are initial values and are expected to be tuned through testing.