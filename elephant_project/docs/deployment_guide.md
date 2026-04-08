# Deployment Guide

## 1. Google Colab
1.  Upload the `elephant_project` folder to Google Drive.
2.  Mount Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/elephant_project
    ```
3.  Install dependencies:
    ```bash
    !pip install ultralytics fastapi uvicorn pyngrok
    ```
4.  Train models using the provided scripts.
5.  Run inference or expose API using `pyngrok`.

## 2. Local Machine (Windows/Linux)
1.  Ensure Python 3.8+ and CUDA (if GPU available) are installed.
2.  Clone/Copy project.
3.  `pip install -r requirements.txt` (Create one with: ultralytics, torch, opencv-python, fastapi, uvicorn).
4.  Run `python src/main.py`.

## 3. Raspberry Pi / Jetson Nano (Edge)
**Jetson Nano (Recommended for GPU support):**
1.  Flash JetPack SDK.
2.  Install PyTorch for Jetson (follow NVIDIA forums).
3.  Install Ultralytics (might need to build from source or use compatible version).
4.  Run `src/main.py`. *Note: Use YOLOv8n (nano) for better FPS.*

**Raspberry Pi 4:**
1.  Install OS (64-bit recommended).
2.  Install TFLite version of YOLOv8 for performance.
3.  Export model: `yolo export model=yolov8n.pt format=tflite`.
4.  Modify `yolo_inference.py` to use TFLite runtime.

## 4. Cloud Inference (FastAPI)
1.  Dockerize the application.
    ```dockerfile
    FROM python:3.9
    WORKDIR /app
    COPY . .
    RUN pip install ultralytics fastapi uvicorn torch opencv-python-headless
    CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
2.  Deploy to AWS EC2 / Google Cloud Run / Azure.
3.  Use GPU instances (e.g., g4dn.xlarge) for real-time performance.

## 5. Mobile SMS Alert (Twilio/Firebase)
- **Twilio**: Sign up, get SID and Token. Update `src/core/alert_system.py`.
- **Firebase**: Create project, download `serviceAccountKey.json`. Use `firebase-admin` SDK in Python.
