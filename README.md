# 🖼️ InstaDetect

Multi-Model Object Detection and Segmentation using OV-DINO & YOLO-World

## 🎯 Overview

InstaDetect is a powerful visual intelligence system that performs open-vocabulary object detection and instance segmentation using state-of-the-art models: OV-DINO and YOLO-World.

## 🔄 Workflow

```mermaid
graph TD
    classDef default fill:#fff,stroke:#333,stroke-width:2px;
    classDef highlight fill:#f9f9f9,stroke:#333,stroke-width:2px;

    A[fa:fa-image Upload Image] --> B{fa:fa-code-branch Model Selection}
    B -->|OV-DINO| C[fa:fa-brain OV-DINO Model]
    B -->|YOLO-World| D[fa:fa-brain YOLO-World Model]
    
    E[fa:fa-keyboard Text Prompt] --> C
    E --> D
    
    C --> F[fa:fa-box-open Object Detection]
    C --> G[fa:fa-puzzle-piece Instance Segmentation]
    D --> F
    
    F --> H[fa:fa-chart-bar Results]
    G --> H
    
    H --> I[fa:fa-eye Visualization]
    H --> J[fa:fa-chart-line Performance Metrics]

    style A fill:#fff,stroke:#333,stroke-width:2px
    style B fill:#fff,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style E fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
    style G fill:#fff,stroke:#333,stroke-width:2px
    style H fill:#fff,stroke:#333,stroke-width:2px
    style I fill:#fff,stroke:#333,stroke-width:2px
    style J fill:#fff,stroke:#333,stroke-width:2px
```

## ✨ Features

- 🔄 Switch between OV-DINO and YOLO-World models
- 📤 Upload images for analysis
- 🎯 Custom text prompt-based detection
- 📦 Bounding box visualization
- 🧩 Instance segmentation
- 📊 Performance metrics display

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/chuck-ankit/instadetect
cd instadetect
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:

Frontend:
```bash
cd app
streamlit run main.py
```

Backend:
```bash
cd backend
uvicorn main:app --reload
```

Note: Run both frontend and backend in separate terminal windows.

## 🏗️ Project Structure

```
instadetect/
├── app/                  # Frontend (Streamlit)
├── backend/             # API backend (FastAPI)
│   ├── routes/
│   ├── ov_dino_engine.py
│   └── yolo_world_engine.py
├── models/              # Model weights and configs
├── utils/              # Helper functions
├── static/             # Image storage
├── requirements.txt
└── README.md
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
