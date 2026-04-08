from ultralytics import YOLO
import cv2
import numpy as np

class PostureDetector:
    def __init__(self, model_path, classifier_path=None):
        """
        Initialize the YOLOv8 Posture Detector.
        
        Args:
            model_path (str): Path to the detection model (e.g., yolov8n.pt).
            classifier_path (str, optional): Path to the custom posture classification model.
        """
        self.detector = YOLO(model_path)
        self.classifier = YOLO(classifier_path) if classifier_path else None
        
        # If no classifier, use detector's classes
        if self.classifier:
            self.classes = self.classifier.names
        else:
            self.classes = self.detector.names

    def predict(self, image):
        """
        Run inference on an image.
        
        Args:
            image (numpy.ndarray): Input image (BGR).
            
        Returns:
            list: List of dictionaries containing bbox, class_id, class_name, confidence, and cropped_img.
        """
        # 1. Detect Elephants
        results = self.detector(image, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter for 'elephant' class if using standard COCO model
                # COCO 'elephant' index is 20
                det_cls_id = int(box.cls[0].cpu().numpy())
                det_cls_name = self.detector.names[det_cls_id]
                
                # If using standard model, only process elephants
                if 'elephant' not in det_cls_name.lower() and self.classifier:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                
                # Crop the elephant
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                cropped_img = image[y1:y2, x1:x2]
                
                final_cls = det_cls_id
                final_name = det_cls_name
                
                # 2. Classify Posture (if classifier exists)
                if self.classifier:
                    cls_results = self.classifier(cropped_img, verbose=False)
                    # Classification result is in probs
                    if cls_results and cls_results[0].probs:
                        probs = cls_results[0].probs
                        top1 = probs.top1
                        final_cls = top1
                        final_name = self.classifier.names[top1]
                        # Update confidence? Maybe combine or keep detection conf?
                        # Let's keep detection confidence for objectness, 
                        # but maybe we want posture confidence.
                        # For now, keep detection confidence as it implies "there is an elephant here"
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': final_cls,
                    'class_name': final_name,
                    'confidence': conf,
                    'cropped_img': cropped_img
                })
        
        return detections

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on the image.
        """
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"Elephant | {det['class_name']} {det['confidence']:.2f}"
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return img_copy

if __name__ == '__main__':
    # Example Usage
    # Replace with actual model path after training
    model_path = 'yolov8m.pt' 
    detector = PostureDetector(model_path)
    
    # Create a dummy image for testing
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (100, 100), (400, 400), (128, 128, 128), -1) # Fake elephant
    
    dets = detector.predict(dummy_img)
    print(f"Detected {len(dets)} objects.")
