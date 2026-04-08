import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from PIL import Image
import torch
from collections import deque
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.yolo_inference import PostureDetector
from src.models.lstm_model import ElephantBehaviourLSTM
from src.core.risk_engine import RiskEngine
from src.core.alert_system import AlertSystem

# Page Config
st.set_page_config(page_title="Elephant Detection System", layout="wide")

st.title("🐘 Elephant Posture & Behaviour Detection")
st.markdown("Real-time AI system for Human-Elephant Conflict mitigation.")

# Sidebar
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Video Upload", "Live Feed", "HEC Heatmap"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

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

# Load Models (Cached)
@st.cache_resource
def load_models():
    # YOLO Standardized to yolov8n.pt for all modes (as requested)
    detector = PostureDetector('yolov8n.pt', 'runs/classify/elephant_posture/weights/best.pt')
    
    # LSTM
    lstm_path = 'src/models/lstm/behaviour_lstm.pth'
    input_size = 10
    hidden_size = 128
    num_layers = 2
    num_classes = 8
    
    behaviour_model = ElephantBehaviourLSTM(input_size, hidden_size, num_layers, num_classes)
    if os.path.exists(lstm_path):
        behaviour_model.load_state_dict(torch.load(lstm_path))
        behaviour_model.eval()
    else:
        behaviour_model = None # Handle missing model
        
    return detector, behaviour_model

detector, behaviour_model = load_models()
risk_engine = RiskEngine()
alert_system = AlertSystem()

# Helper to process frame with tracking
def process_frame(frame, detector, behaviour_model, tracker, track_states):
    # Posture Detection
    detections = detector.predict(frame)
    
    # Filter by confidence
    if detections:
        detections = [d for d in detections if d['confidence'] >= confidence_threshold]
    
    # Format for Tracker
    tracker_inputs = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        h = y2 - y1
        conf = det['confidence']
        cls_id = det['class_id']
        tracker_inputs.append([x1, y1, w, h, conf, cls_id])
        
    # Update Tracker
    tracks = tracker.update(tracker_inputs, frame)
    
    annotated_frame = frame.copy()
    
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # Match detection for posture
        current_posture_name = "Unknown"
        current_posture_cls = 0
        
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
        
        if best_det and min_dist < 5000:
            current_posture_cls = best_det['class_id']
            raw_posture_name = best_det['class_name']
            current_posture_name = POSTURE_MAP.get(raw_posture_name.lower(), raw_posture_name)
            
        # Initialize state if new track
        if track_id not in track_states:
            dx, dy = 0, 0
            # Features for initial buffer
            features = [current_posture_cls, x1, y1, x2, y2, dx, dy, 0, 0, 0]
            
            # Pre-fill buffer with current features to allow immediate prediction
            initial_buffer = deque([features] * 30, maxlen=30)
            track_states[track_id] = {
                'buffer': initial_buffer,
                'prev_center': track_center, # Set prev_center to current to avoid jump on next frame
                'behaviour': "",
                'risk': "Low"
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
            
            # Features
            features = [current_posture_cls, x1, y1, x2, y2, dx, dy, 0, 0, 0]
            state['buffer'].append(features)
            state['prev_center'] = track_center
        
        # Predict Behaviour
        if behaviour_model and len(state['buffer']) == 30:
            seq_tensor = torch.tensor([list(state['buffer'])], dtype=torch.float32)
            with torch.no_grad():
                beh_logits, risk_prob, alert_prob = behaviour_model(seq_tensor)
                beh_idx = torch.argmax(beh_logits, dim=1).item()
                # Updated to match user dataset labels
                beh_labels = ['Aggressive', 'Alert', 'Calm', 'Distress', 'Feeding', 'Migration', 'Social Interaction', 'Warning Display']
                if beh_idx < len(beh_labels):
                    state['behaviour'] = beh_labels[beh_idx]
                
                alert_score = alert_prob.item()
                if alert_score > 0.7:
                    state['behaviour'] += " (Aggressive)"
                elif alert_score < 0.3:
                    state['behaviour'] += " (Calm)"
        
        # Risk
        state['risk'] = risk_engine.evaluate_risk(state['behaviour'], current_posture_name)
        
        if state['risk'] in ["High", "Medium"]:
            # Add explicit tag for Aggressive Behaviour in Simulation (per user request)
            is_aggressive = "Aggressive" in state['behaviour']
            alert_tag = "AGGRESSIVE_BEHAVIOUR" if is_aggressive else state['risk']
            alert_prefix = "🚨 EXTREME DANGER (AGGRESSIVE BEHAVIOUR)" if is_aggressive else f"{state['risk']} Risk"
            
            alert_dispatched = alert_system.send_alert(
                risk_level=alert_tag, 
                details=f"{alert_prefix} | ID {track_id}: {state['behaviour']} elephant ({current_posture_name}) detected in simulation!", 
                location="Simulation / Streamlit Video Feed"
            )
            if alert_dispatched and hasattr(st, 'toast'):
                st.toast(f"🚨 ALERT SENT TO TELEGRAM: {alert_prefix} ({current_posture_name})", icon="⚠️")

        
        # Draw
        color = (0, 255, 0)
        if state['risk'] == "High": color = (0, 0, 255)
        elif state['risk'] == "Medium": color = (255, 165, 0)
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"Elephant ID: {track_id} | {current_posture_name}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        beh_text = f"Behaviour: {state['behaviour']} | " if state['behaviour'] else ""
        cv2.putText(annotated_frame, f"{beh_text}Risk: {state['risk']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Group Detection Logic
    active_tracks = [t for t in tracks if t.is_confirmed()]
    if len(active_tracks) > 3:
        # Calculate group bounding box
        min_x = min([int(t.to_ltrb()[0]) for t in active_tracks])
        min_y = min([int(t.to_ltrb()[1]) for t in active_tracks])
        max_x = max([int(t.to_ltrb()[2]) for t in active_tracks])
        max_y = max([int(t.to_ltrb()[3]) for t in active_tracks])
        
        # Draw Group Box (Purple)
        cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), (255, 0, 255), 3)
        cv2.putText(annotated_frame, "Elephant Group", (min_x, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

    return annotated_frame

# --- Modes ---

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Image mode doesn't support tracking well, just run detection
        # Consolidated to use the same detector (yolov8n.pt) as feeds
        detections = detector.predict(img_bgr)
        
        # Apply Posture Mapping for Image mode
        for det in detections:
            raw_name = det['class_name']
            det['class_name'] = POSTURE_MAP.get(raw_name.lower(), raw_name)
            
        annotated_img = detector.draw_detections(img_bgr, detections)
        
        # Group Detection for Image
        if len(detections) > 3:
            min_x = min([d['bbox'][0] for d in detections])
            min_y = min([d['bbox'][1] for d in detections])
            max_x = max([d['bbox'][2] for d in detections])
            max_y = max([d['bbox'][3] for d in detections])
            
            cv2.rectangle(annotated_img, (min_x, min_y), (max_x, max_y), (255, 0, 255), 3)
            cv2.putText(annotated_img, "Elephant Group", (min_x, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Processed Image", use_column_width=True)

elif mode == "Video Upload":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        # Init Tracker
        from src.core.tracker import ElephantTracker
        tracker = ElephantTracker()
        track_states = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame, detector, behaviour_model, tracker, track_states)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")
            
        cap.release()

elif mode == "Live Feed":
    st.write("Starting Webcam...")
    run = st.checkbox('Run Feed')
    st_frame = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        # Init Tracker
        from src.core.tracker import ElephantTracker
        tracker = ElephantTracker()
        track_states = {}
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
            processed_frame = process_frame(frame, detector, behaviour_model, tracker, track_states)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")
        
        cap.release()

# --- HEC Heatmap Mode ---

elif mode == "HEC Heatmap":
    st.header("🗺️ Human-Elephant Conflict Heatmap (South India)")
    st.write("Visualizing past elephant corridors and high-risk conflict zones to predict future encounters.")

    # Dummy data for demonstration based on South India High Conflict zones
    data = [
        {"name": "Wayanad, Kerala", "lat": 11.6234, "lon": 76.1415, "weight": 0.8},
        {"name": "Bandipur, Karnataka", "lat": 11.6601, "lon": 76.5610, "weight": 0.9},
        {"name": "Mudumalai, Tamil Nadu", "lat": 11.4916, "lon": 76.7337, "weight": 0.6},
        {"name": "Nagarhole, Karnataka", "lat": 11.9362, "lon": 76.2570, "weight": 0.8},
        {"name": "Anamalai, Tamil Nadu", "lat": 10.4526, "lon": 77.0375, "weight": 0.5},
        {"name": "Munnar (Idukki), Kerala", "lat": 10.0889, "lon": 77.0595, "weight": 0.85},
        {"name": "Kodagu (Coorg), Karnataka", "lat": 12.3375, "lon": 75.8069, "weight": 0.75},
        {"name": "Hassan, Karnataka", "lat": 13.0033, "lon": 76.1004, "weight": 0.9},
        {"name": "Coimbatore/Valparai, TN", "lat": 11.0168, "lon": 76.9558, "weight": 0.7},
        {"name": "Sathyamangalam (Erode), TN", "lat": 11.3424, "lon": 77.7172, "weight": 0.65},
        {"name": "Palakkad, Kerala", "lat": 10.7867, "lon": 76.6548, "weight": 0.8},
        {"name": "Nilambur, Kerala", "lat": 11.2750, "lon": 76.2250, "weight": 0.6},
        {"name": "Chamarajanagar, Karnataka", "lat": 11.9261, "lon": 76.9400, "weight": 0.7},
        {"name": "Gudalur, Tamil Nadu", "lat": 11.5033, "lon": 76.5050, "weight": 0.7},
        {"name": "Dharmapuri (Hosur), TN", "lat": 12.1200, "lon": 78.1600, "weight": 0.5},
        {"name": "BR Hills, Karnataka", "lat": 11.9600, "lon": 77.1400, "weight": 0.8},
        {"name": "Sakleshpur, Karnataka", "lat": 12.9722, "lon": 75.7864, "weight": 0.85},
        {"name": "Attappadi, Kerala", "lat": 11.1000, "lon": 76.6000, "weight": 0.9},
        {"name": "Kodaikanal, Tamil Nadu", "lat": 10.2381, "lon": 77.4892, "weight": 0.4},
        {"name": "Megamalai, Tamil Nadu", "lat": 10.2600, "lon": 77.3800, "weight": 0.6},
        {"name": "Silent Valley, Kerala", "lat": 11.1200, "lon": 76.4300, "weight": 0.75},
        {"name": "Thenmala, Kerala", "lat": 8.9650,  "lon": 77.0600, "weight": 0.65},
        {"name": "Sullia, Karnataka", "lat": 12.5600, "lon": 75.3900, "weight": 0.7},
    ]
    df = pd.DataFrame(data)

    # Create Base Map centered on South India
    m = folium.Map(location=[11.5, 76.5], zoom_start=8, tiles="CartoDB dark_matter")

    # Add Heatmap Layer
    heat_data = [[row['lat'], row['lon'], row['weight']] for index, row in df.iterrows()]
    HeatMap(heat_data, radius=25, blur=15, max_zoom=1).add_to(m)

    # Add Clickable/Hoverable Markers
    for index, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color='yellow',
            fill=True,
            fill_opacity=0.4,
            tooltip=f"{row['name']} (Lat: {row['lat']}, Lon: {row['lon']})"
        ).add_to(m)

    # Render in Streamlit
    st_folium(m, width=800, height=500, returned_objects=[])

    with st.expander("📍 View & Search All Coordinate Regions Dataset", expanded=False):
        st.dataframe(df[['name', 'lat', 'lon']], use_container_width=True)
    
    st.subheader("🔮 Conflict Prediction Engine")
    cols = st.columns(2)
    with cols[0]:
        selected_village_lat = st.number_input("Enter Latitude", value=11.6230, format="%.4f")
        selected_village_lon = st.number_input("Enter Longitude", value=76.1410, format="%.4f")
    with cols[1]:
        selected_month = st.selectbox("Select Upcoming Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        crop_type = st.selectbox("Current Crop in Area", ["Paddy", "Jackfruit", "Banana", "Tea/Coffee", "None"])

    if st.button("Predict Arrival Probability"):
        prob = 30
        if crop_type in ["Paddy", "Jackfruit"]: prob += 40
        if selected_month in ["June", "July", "August", "September"]: prob += 20
        
        if prob > 70:
            msg = f"⚠️ **{prob}% Probability** of elephant herd arrival in this sector due to monsoon and {crop_type} harvesting."
            st.error(msg)
            # Dispatch early warning to Telegram
            alert_system.send_alert(
                "High", 
                f"EARLY WARNING: {prob}% probability of elephant herd arrival. Reason: Monsoon active & {crop_type} harvest.", 
                location=f"Lat {selected_village_lat:.4f}, Lon {selected_village_lon:.4f}"
            )
        elif prob > 40:
            st.warning(f"🟡 **{prob}% Probability** of elephant movement. Stay alert.")
        else:
            st.success(f"🟢 **{prob}% Probability**. Low risk of conflict currently.")

