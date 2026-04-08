import os
import cv2
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path to import core modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from core.yolo_inference import PostureDetector

def process_videos(source_dir, output_csv, skip_frames=5):
    # Initialize Posture Detector
    # Using the trained classifier and standard YOLO for detection
    classifier_path = 'runs/classify/elephant_posture/weights/best.pt'
    if not os.path.exists(classifier_path):
        print(f"Warning: Classifier not found at {classifier_path}. Posture labels will be generic.")
        classifier_path = None
        
    detector = PostureDetector('yolov8m.pt', classifier_path)
    
    source_path = Path(source_dir)
    output_path = Path(output_csv)
    
    # Check for existing progress
    processed_videos = set()
    next_video_id = 1
    
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if 'video_file' in existing_df.columns:
                processed_videos = set(existing_df['video_file'].unique())
                print(f"Found {len(processed_videos)} already processed videos.")
            
            if 'elephant_id' in existing_df.columns:
                 # Resume ID counter
                 next_video_id = existing_df['elephant_id'].max() + 1
                 
        except Exception as e:
            print(f"Could not read existing CSV: {e}")
    
    # Iterate over behaviour classes
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        behaviour_label = class_dir.name
        
        # Map behaviour to risk/alertness
        conflict_risk = 'Low'
        alertness_label = 'Calm'
        if behaviour_label in ['aggressive', 'charging', 'warning display']:
            conflict_risk = 'High'
            alertness_label = 'Aggressive'
        elif behaviour_label in ['alert', 'distress']:
            conflict_risk = 'Medium'
            alertness_label = 'Alert'
            
        for video_file in class_dir.iterdir():
            if video_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
                continue
                
            if video_file.name in processed_videos:
                continue

            print(f"Processing: {behaviour_label}/{video_file.name} (ID: {next_video_id})")
            cap = cv2.VideoCapture(str(video_file))
            
            video_data = []
            frame_count = 0
            
            prev_center = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue
                
                # Run inference
                detections = detector.predict(frame)
                
                # We assume the largest elephant is the subject of the video
                if detections:
                    det = max(detections, key=lambda x: x['confidence'])
                    x1, y1, x2, y2 = det['bbox']
                    posture_cls = det['class_id']
                    # posture_name = det['class_name'] 
                    
                    center = ((x1+x2)/2, (y1+y2)/2)
                    
                    move_dx, move_dy = 0.0, 0.0
                    if prev_center is not None:
                        move_dx = center[0] - prev_center[0]
                        move_dy = center[1] - prev_center[1]
                    
                    prev_center = center
                    
                    # Placeholder for specific features not yet implemented
                    trunk_angle = 0.0
                    ear_freq = 0.0
                    tail_freq = 0.0
                    
                    row = {
                        'frame_id': frame_count,
                        'elephant_id': next_video_id, # Unique ID per video
                        'posture_class': posture_cls,
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'move_dx': float(move_dx),
                        'move_dy': float(move_dy),
                        'trunk_angle': float(trunk_angle),
                        'ear_freq': float(ear_freq),
                        'tail_freq': float(tail_freq),
                        'behaviour_label': behaviour_label,
                        'conflict_risk': conflict_risk,
                        'alertness_label': alertness_label,
                        'video_file': video_file.name
                    }
                    video_data.append(row)
            
            cap.release()
            
            # Save incrementally
            if video_data:
                df_chunk = pd.DataFrame(video_data)
                header = not output_path.exists()
                df_chunk.to_csv(output_path, mode='a', header=header, index=False)
            
            next_video_id += 1

if __name__ == "__main__":
    SOURCE_DIR = r"c:\elephant posture and behaviour\elephant_project\elephant_dataset\behaviour"
    OUTPUT_CSV = r"c:\elephant posture and behaviour\elephant_project\behaviour_dataset_extracted.csv"
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory {SOURCE_DIR} not found.")
    else:
        print("Starting video processing...")
        process_videos(SOURCE_DIR, OUTPUT_CSV, skip_frames=5)
        print("Processing complete.")
