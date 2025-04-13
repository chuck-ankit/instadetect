from transformers import AutoModelForObjectDetection, AutoProcessor
import torch
import numpy as np
from PIL import Image

class OVDINODetector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        if self.model is None:
            self.model = AutoModelForObjectDetection.from_pretrained(
                "ShilongLiu/ov-dino-base",
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained("ShilongLiu/ov-dino-base")
    
    def detect(self, image: Image.Image, prompts: list, confidence_threshold: float = 0.5):
        self.load_model()
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=prompts,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "box": box.tolist(),
                "score": score.item(),
                "label": prompts[label],
            })
        
        return detections
