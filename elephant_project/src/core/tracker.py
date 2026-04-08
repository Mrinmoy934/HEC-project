from deep_sort_realtime.deepsort_tracker import DeepSort

class ElephantTracker:
    def __init__(self, max_age=30, n_init=3):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep a track alive without new detections.
            n_init (int): Number of consecutive detections to confirm a track.
        """
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        
    def update(self, detections, frame):
        """
        Update tracks with new detections.
        
        Args:
            detections (list): List of detections in format [[x1, y1, w, h, score, class_id], ...]
            frame (numpy.ndarray): Current video frame (required for DeepSORT feature extraction).
            
        Returns:
            list: List of tracks. Each track object has .track_id and .to_ltwh().
        """
        # DeepSORT expects detections as (ltwh, confidence, class_id)
        # Our input is [x1, y1, w, h, score, class_id]
        
        formatted_detections = []
        for det in detections:
            bbox = det[:4] # x1, y1, w, h
            # Convert x1,y1,w,h to left,top,w,h (which it already is if x1 is left)
            # YOLO usually gives center_x, center_y, w, h OR x1, y1, x2, y2.
            # We need to ensure what we pass in.
            # Let's assume the caller passes [left, top, w, h]
            
            conf = det[4]
            cls_id = det[5]
            formatted_detections.append((bbox, conf, cls_id))
            
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        
        return tracks
