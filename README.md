# Micro-Expression Observation System

A non-invasive, computer-vision based system for observing and reporting facial micro-expressions.

## Overview

This project uses **MediaPipe Face Mesh** to track key facial regions (eyebrows, eyes, lips) and detects "events" where facial movements exceed a calculated baseline. It uses a **Random Forest Classifier** (trained on the Micro-Expressions Emotion Intensity Dataset) to predict the emotional context (e.g., happiness, surprise, neutral) of these events.

**Privacy Note**: All processing is performed locally. No video or biometric data is sent to the cloud.

## Features

- **Real-time Tracking**: 468 facial landmarks at 30+ FPS.
- **Event Detection**: Detects distinct actions (e.g., Brow Raise, Frown) using Z-score outlier detection.
- **Emotion Classification**: Predicts emotional context for detected events.
- **Detailed Reporting**: Generates a timestamped text report of all observations.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the main script:
    ```bash
    python main.py
    ```
2.  **Calibration**: A webcam window will open. Keep your face still and neutral for **3 seconds**.
3.  **Monitoring**: The status will change to "Monitoring".
4.  **Action**: Perform facial expressions. Events will be logged in the console.
5.  **Finish**: Press `q` to quit.
6.  **Report**: Open `report.txt` to view the session summary.

## Files

- `main.py`: Entry point. Runs the webcam loop.
- `detector.py`: Event detection logic and state machine.
- `feature_extraction.py`: Maps landmarks to Action Unit proxies.
- `model_trainer.py`: Script used to train `emotion_model.pkl`.
- `report_generator.py`: Formats the final text report.

## Disclaimer

This tool is for observational research and engineering demonstration purposes only. It is **not** a diagnostic tool and should not be used for medical or clinical assessment.
