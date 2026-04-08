# Elephant Posture and Behaviour Detection System

## Abstract
This project implements a real-time system for detecting elephant postures and behaviours to mitigate Human-Elephant Conflict (HEC). It utilizes YOLOv8 for object and posture detection and an LSTM model for temporal behaviour analysis.

## Methodology
1.  **Posture Detection**: YOLOv8m trained on 16 posture classes.
2.  **Behaviour Detection**: LSTM network processing sequences of posture and movement features.
3.  **Risk Assessment**: Rule-based engine combining behaviour and posture to determine risk levels (High, Medium, Low).
4.  **Alert System**: Automated SMS/Firebase notifications for high-risk events.

## Architecture
- **Input**: Video Stream / Camera.
- **Core**: YOLOv8 -> Feature Extraction -> LSTM -> Risk Engine.
- **Output**: Visual Overlay + API Alerts.

## Directory Structure
- `data/`: Dataset storage.
- `models/`: Trained .pt and .pth models.
- `src/`: Source code (API, Core, Training).

## Installation
1.  Install dependencies:
    ```bash
    pip install ultralytics torch opencv-python fastapi uvicorn pandas
    ```
2.  Prepare Dataset in `data/`.
3.  Train YOLO: `python src/training/train_yolo.py`
4.  Train LSTM: `python src/training/train_behaviour.py`

## Usage
- **Run End-to-End**:
    ```bash
    python src/main.py
    ```
- **Run API**:
    ```bash
    uvicorn src.api.app:app --reload
    ```

## Results
- **Posture mAP**: TBD (Target > 0.8)
- **Behaviour Accuracy**: TBD (Target > 0.85)

## Future Scope
- Integration with Thermal Cameras for night vision.
- Edge deployment on Jetson Nano.
- Audio distress call integration.
