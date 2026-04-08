import cv2
import torch
import numpy as np
from collections import deque
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from core.yolo_inference import PostureDetector
from models.lstm_model import ElephantBehaviourLSTM
from core.risk_engine import RiskEngine
from core.alert_system import AlertSystem

def main(source=0, detector_path='yolov8n.pt', classifier_path='runs/classify/elephant_posture/weights/best.pt', lstm_path='src/models/lstm/behaviour_lstm.pth'):
    """
    Main loop for Real-time Elephant Posture and Behaviour Detection.
    """
    # 1. Load Models
    print("Loading models...")
    
    try:
        posture_detector = PostureDetector(detector_path, classifier_path)
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return

    # User-defined Class Mapping
    POSTURE_MAP = {
        'calf under mother': 'calf',
        'charging': 'charging',
        'drinking': 'water drinking',
        'dusting': 'dusting',
        'ear flapping': 'ear flapping',
        'eating': 'eating',
        'herd': 'herd',
        'mud bathing': 'mud bathing',
        'running': 'running',
        'sleeping': 'sleeping',
        'standing': 'elephant',
        'tail swing': 'tail swinging',
        'trunk down': 'Trunk Down',
        'trunk up': 'trunk up',
        'walking': 'elephant walking',
        'group': 'herd',
        'unknown': 'elephant'
    }

    # Tracker Setup
    from core.tracker import ElephantTracker
    tracker = ElephantTracker(max_age=30, n_init=3)

    # LSTM Setup
    input_size = 10 # Must match training
    hidden_size = 128
    num_layers = 2
    num_classes = 8
    
    behaviour_model = ElephantBehaviourLSTM(input_size, hidden_size, num_layers, num_classes)
    if os.path.exists(lstm_path):
        behaviour_model.load_state_dict(torch.load(lstm_path))
        behaviour_model.eval()
    else:
        print("Warning: LSTM model not found. Behaviour prediction will be random/untrained.")
    
    risk_engine = RiskEngine()
    alert_system = AlertSystem()
    
    # 2. Video Capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # State Management for Multiple Tracks
    # Dictionary: track_id -> { 'buffer': deque, 'prev_center': (x,y), 'behaviour': str, 'risk': str }
    track_states = {}
    sequence_length = 30
    
    print("Starting inference loop. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Posture Detection
        detections = posture_detector.predict(frame)
        
        # Format detections for Tracker: [left, top, w, h, confidence, class_id]
        tracker_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            conf = det['confidence']
            cls_id = det['class_id'] # Posture Class
            tracker_inputs.append([x1, y1, w, h, conf, cls_id])
            
        # 4. Update Tracker
        tracks = tracker.update(tracker_inputs, frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Get posture class from the track (DeepSORT keeps the class ID)
            # Note: deep-sort-realtime stores original detection info in track.det_class if configured, 
            # or we pass it through. The library usually keeps the class_id passed in update.
            posture_cls = track.det_class if hasattr(track, 'det_class') else 0
            # If det_class is missing or None, we might need to map it back or just use the latest detection.
            # For simplicity, let's assume the tracker maintains it. 
            # Actually, deep-sort-realtime might not store custom attributes by default easily.
            # A robust way is to match track bbox with detection bbox, but let's trust the tracker's class_id if available.
            # If not, we default to 0 (Standing) or try to find the matching detection.
            
            # Let's try to find the closest detection to this track to get the fresh posture
            # (since posture changes frame to frame, but track ID stays)
            current_posture_cls = 0
            current_posture_name = "Unknown"
            
            # Simple IoU or center distance check to find matching detection
            track_center = ((x1+x2)/2, (y1+y2)/2)
            best_det = None
            min_dist = float('inf')
            
            for det in detections:
                d_x1, d_y1, d_x2, d_y2 = det['bbox']
                d_center = ((d_x1+d_x2)/2, (d_y1+d_y2)/2)
                dist = (track_center[0]-d_center[0])**2 + (track_center[1]-d_center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_det = det
            
            if best_det and min_dist < 5000: # Threshold
                current_posture_cls = best_det['class_id']
                raw_posture_name = best_det['class_name']
                current_posture_name = POSTURE_MAP.get(raw_posture_name.lower(), raw_posture_name)
            
            # Initialize state if new track
            if track_id not in track_states:
                dx, dy = 0, 0
                trunk_angle = 0 # Placeholder
                ear_freq = 0
                tail_freq = 0
                
                features = [current_posture_cls, x1, y1, x2, y2, dx, dy, trunk_angle, ear_freq, tail_freq]
                
                # Pre-fill buffer
                initial_buffer = deque([features] * sequence_length, maxlen=sequence_length)
                track_states[track_id] = {
                    'buffer': initial_buffer,
                    'prev_center': track_center,
                    'behaviour': "",
                    'risk': "Low",
                    'alert_score': 0.0
                }
                state = track_states[track_id]
            else:
                state = track_states[track_id]
                # Calculate movement
                prev_center = state['prev_center']
                if prev_center:
                    dx = track_center[0] - prev_center[0]
                    dy = track_center[1] - prev_center[1]
                else:
                    dx, dy = 0, 0
                
                trunk_angle = 0 # Placeholder
                ear_freq = 0
                tail_freq = 0
                
                features = [current_posture_cls, x1, y1, x2, y2, dx, dy, trunk_angle, ear_freq, tail_freq]
                state['buffer'].append(features)
                state['prev_center'] = track_center
            
            # Predict Behaviour
            if len(state['buffer']) == sequence_length:
                seq_tensor = torch.tensor([list(state['buffer'])], dtype=torch.float32)
                with torch.no_grad():
                    beh_logits, risk_prob, alert_prob = behaviour_model(seq_tensor)
                    beh_idx = torch.argmax(beh_logits, dim=1).item()
                    # Updated to match user dataset labels
                    beh_labels = ['Aggressive', 'Alert', 'Calm', 'Distress', 'Feeding', 'Migration', 'Social Interaction', 'Warning Display']
                    if beh_idx < len(beh_labels):
                        state['behaviour'] = beh_labels[beh_idx]
                    
                    state['alert_score'] = alert_prob.item()
                    
                    if state['alert_score'] > 0.7:
                        state['behaviour'] += " (Aggressive)"
                    elif state['alert_score'] < 0.3:
                        state['behaviour'] += " (Calm)"
            
            # Risk
            state['risk'] = risk_engine.evaluate_risk(state['behaviour'], current_posture_name)
            
            # Alert
            if state['risk'] in ["High", "Medium"]:
                alert_system.send_alert(state['risk'], f"ID {track_id}: {state['behaviour']} elephant detected!")
            
            # Visualization
            color = (0, 255, 0)
            if state['risk'] == "High": color = (0, 0, 255)
            elif state['risk'] == "Medium": color = (0, 165, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            beh_text = f" | Behaviour: {state['behaviour']}" if state['behaviour'] else ""
            label_text = f"Elephant ID: {track_id} | {current_posture_name}{beh_text}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Group Detection Logic
        active_tracks = [t for t in tracks if t.is_confirmed()]
        if len(active_tracks) > 3:
            # Calculate group bounding box
            min_x = min([int(t.to_ltrb()[0]) for t in active_tracks])
            min_y = min([int(t.to_ltrb()[1]) for t in active_tracks])
            max_x = max([int(t.to_ltrb()[2]) for t in active_tracks])
            max_y = max([int(t.to_ltrb()[3]) for t in active_tracks])
            
            # Draw Group Box (Purple)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 255), 3)
            cv2.putText(frame, "Elephant Group", (min_x, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
            
        cv2.imshow('Elephant Detection System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure directories exist for models
    # This script assumes it's run from the project root or src
    # Adjust paths as needed
    main()
