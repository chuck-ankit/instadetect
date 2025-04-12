# InstaDetect - Multi-Model Object Detection

This Hugging Face Space hosts InstaDetect, an interactive object detection system that uses OV-DINO and YOLO-World models for detecting objects in images.

## Features

- 🔄 Switch between OV-DINO and YOLO-World models
- 📤 Upload images for analysis
- 🎯 Custom text prompt-based detection
- 📦 Bounding box visualization
- 📊 Detailed performance metrics

## How to Use

1. Select a detection model (OV-DINO or YOLO-World)
2. Enter detection prompts or use the default ones
3. Upload an image
4. View detailed detection results

## Models

- **OV-DINO**: Open-vocabulary object detection based on DETR architecture
- **YOLO-World**: Open-vocabulary version of YOLOv8

## Technical Details

- Frontend: Streamlit
- Backend: FastAPI
- Image Processing: PIL, OpenCV
- Deep Learning: PyTorch, Transformers

## Acknowledgments

- OV-DINO: [ShilongLiu/ov-dino](https://huggingface.co/ShilongLiu/ov-dino)
- YOLO-World: [Ultralytics](https://github.com/ultralytics/ultralytics)

---
tags: [object-detection, computer-vision, streamlit, fastapi]
