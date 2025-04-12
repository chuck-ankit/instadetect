from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from pydantic import BaseModel
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import time
import random

app = FastAPI(title="InstaDetect API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock detection function
def mock_detect(image: Image.Image, prompts: List[str], confidence_threshold: float = 0.5):
    width, height = image.size
    detections = []
    
    # Create mock detections for each prompt
    for prompt in prompts:
        # Create 1-3 detections for each prompt
        for _ in range(random.randint(1, 3)):
            # Random box coordinates
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            x2 = min(x1 + random.randint(50, 200), width)
            y2 = min(y1 + random.randint(50, 200), height)
            
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": random.uniform(confidence_threshold, 1.0),
                "label": prompt
            })
    
    return detections

# Mock visualization function
def draw_detections(image: Image.Image, detections: List[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    for det in detections:
        box = det["box"]
        label = f"{det['label']} {det['score']:.2f}"
        
        # Draw box
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label
        draw.text((box[0], box[1] - 10), label, fill="red")
    
    return result

class DetectionRequest(BaseModel):
    prompts: List[str]
    confidence_threshold: float
    model_name: str

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    confidence_threshold: float = Form(0.5),
    prompts: str = Form(...)
):
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert prompts string to list
        prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]
        
        # Start timing
        start_time = time.time()
        
        # Run mock detection
        detections = mock_detect(image, prompt_list, float(confidence_threshold))
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Draw detections
        result_image = draw_detections(image, detections)
        
        # Convert result image to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "message": "Detection completed",
            "model_used": model_name,
            "detections": detections,
            "inference_time": inference_time,
            "image_base64": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
