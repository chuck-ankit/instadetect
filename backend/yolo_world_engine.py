from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image

class YOLOWorldDetector:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is None:
            self.model = YOLO('yolo_world.pt')  # Download from official YOLO-World repository
    
    def detect(self, image: Image.Image, prompts: list, confidence_threshold: float = 0.5):
        self.load_model()
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Run inference with YOLO-World
        results = self.model.predict(
            source=image_np,
            classes=prompts,
            conf=confidence_threshold,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            detections.append({
                "box": box.xyxy[0].tolist(),
                "score": box.conf.item(),
                "label": prompts[box.cls.item()],
            })
        
        return detections
