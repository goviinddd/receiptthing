from ultralytics import YOLO
from PIL import Image
import numpy as np

class TableDetector:
    def __init__(self, model_path, device="cuda"):
        print(f"[Detector] Loading YOLOv8 from {model_path}...")
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image: Image.Image, conf=0.1):
        """
        Returns: List of detected objects with metadata
        """
        results = self.model(image, conf=conf, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            coords = map(int, box.xyxy[0].cpu().numpy())
            detections.append({
                "class_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "bbox": list(coords) # [x1, y1, x2, y2]
            })
        return detections