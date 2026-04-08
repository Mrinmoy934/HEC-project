from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import cv2
import numpy as np
import sys

# Add src to path
sys.path.append(os.path.abspath('..'))

from core.yolo_inference import PostureDetector
from core.risk_engine import RiskEngine
from core.alert_system import AlertSystem

app = FastAPI(title="Elephant Detection API", version="1.0")

# Initialize modules (Mock paths for now)
# In production, these would be real paths to trained models
MODEL_PATH = '../../models/yolo/elephant_posture_v8/weights/best.pt' 
# If model doesn't exist, we might need a fallback or lazy loading
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model not found at {MODEL_PATH}. Inference will fail.")
    detector = None
else:
    detector = PostureDetector(MODEL_PATH)

risk_engine = RiskEngine()
alert_system = AlertSystem()

class PredictionResponse(BaseModel):
    posture: str
    behaviour: str
    risk: str
    confidence: float

@app.get("/")
def home():
    return {"message": "Elephant Detection System API is Running"}

@app.post("/predict_posture")
async def predict_posture(file: UploadFile = File(...)):
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Save temp file
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Read image
    img = cv2.imread(temp_file)
    if img is None:
        os.remove(temp_file)
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    # Inference
    detections = detector.predict(img)
    
    # Cleanup
    os.remove(temp_file)
    
    return {"detections": detections}

@app.post("/alert")
async def trigger_alert(risk: str, details: str):
    """
    Manual endpoint to trigger alerts for testing.
    """
    alert_system.send_alert(risk, details)
    return {"status": "Alert processed"}

# Note: Full behaviour prediction requires a sequence of frames, 
# which is hard to do via a simple single-file upload endpoint.
# Usually, this would process a video stream or accept a batch of features.
