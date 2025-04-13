import cv2
import numpy as np
from PIL import Image
import supervision as sv

class DetectionVisualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
    
    def draw_detections(self, image: Image.Image, detections: list) -> Image.Image:
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Convert detections to supervision format
        boxes = []
        labels = []
        for det in detections:
            boxes.append(det["box"])
            labels.append(f"{det['label']} {det['score']:.2f}")
        
        # Convert to supervision detections
        detections = sv.Detections(
            xyxy=np.array(boxes),
            class_id=np.arange(len(boxes)),
        )
        
        # Draw annotations
        frame = self.box_annotator.annotate(
            scene=image_np,
            detections=detections,
            labels=labels
        )
        
        # Convert back to PIL Image
        return Image.fromarray(frame)
