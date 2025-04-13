import streamlit as st
import requests
from PIL import Image
import io
import os
import json
import time

st.set_page_config(
    page_title="InstaDetect",
    page_icon="🖼️",
    layout="wide"
)

def main():
    st.title("🖼️ InstaDetect")
    st.subheader("Multi-Model Object Detection and Segmentation")

    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("Settings")
        model_choice = st.radio(
            "Select Detection Model",
            ["OV-DINO", "YOLO-World"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Default comprehensive prompts
        default_prompts = """
person
car
dog
cat
chair
table
couch
bed
laptop
cell phone
backpack
bicycle
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
handbag
umbrellas
shoes
eye glasses
hat
book
clock
vase
scissors
toothbrush
hair brush
keyboard
mouse
remote control
microwave
refrigerator
washing machine"""

        # Custom prompts with default values
        custom_prompts = st.text_area(
            "Detection Prompts",
            value=default_prompts,
            help="Edit or add objects to detect (one per line). These are common objects that the model can detect.",
            height=400
        )
        
        # Process prompts
        if custom_prompts:
            prompts_list = [p.strip() for p in custom_prompts.split('\n') if p.strip()]
            if prompts_list:
                st.markdown("### Selected Objects to Detect:")
                st.write(f"Total objects: {len(prompts_list)}")
                st.write(", ".join(prompts_list))
            
            # Store prompts in session state
            st.session_state.detection_prompts = custom_prompts

    # Initialize session state for storing results
    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = None

    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image_source = st.radio(
            "Select Image Source",
            ["Upload Image", "Use Camera"],
            horizontal=True
        )
        
        uploaded_image = None
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_image = uploaded_file
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            camera_input = st.camera_input("Take a picture")
            if camera_input is not None:
                uploaded_image = camera_input
                image = Image.open(camera_input)
                st.image(image, caption="Captured Image", use_column_width=True)
        
        # Add detect button
        if uploaded_image is not None and st.session_state.get("detection_prompts"):
            if st.button("Detect Objects", type="primary"):
                with st.spinner("Detecting objects..."):
                    try:
                        # Prepare the files and data for the API request
                        files = {
                            'file': ('image.jpg', uploaded_image.getvalue(), 'image/jpeg')
                        }
                        data = {
                            'model_name': model_choice,
                            'confidence_threshold': str(confidence_threshold),
                            'prompts': st.session_state.detection_prompts
                        }
                        
                        # Make API request to the backend
                        response = requests.post(
                            'http://localhost:8000/detect',
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.detection_results = result
                            # The processed image will be shown in the right column
                            st.experimental_rerun()
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to the backend server: {str(e)}")
        elif uploaded_image is None:
            st.info("Please upload an image or take a picture")
        elif not st.session_state.get("detection_prompts"):
            st.info("Please enter detection prompts in the sidebar")

    with col2:
        st.subheader("Detection Results")
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            # Display detection results
            st.image(results.get("image_base64"), caption="Detection Results", use_column_width=True)
            
            if results.get('detections'):
                # Group detections by label
                detections_by_label = {}
                for det in results['detections']:
                    label = det['label']
                    if label not in detections_by_label:
                        detections_by_label[label] = []
                    detections_by_label[label].append(det['score'])
                
                # Display summary metrics
                st.markdown("### Summary")
                total_objects = len(results.get('detections', []))
                unique_objects = len(detections_by_label)
                st.markdown(f"""
                - **Total Objects Detected**: {total_objects}
                - **Unique Object Types**: {unique_objects}
                - **Model Used**: {model_choice}
                - **Inference Time**: {results.get('inference_time', 0):.2f}ms
                """)
                
                # Display detailed results by category
                st.markdown("### Detailed Results")
                for label, scores in detections_by_label.items():
                    with st.expander(f"{label.title()} ({len(scores)} instances)"):
                        # Show average confidence
                        avg_conf = sum(scores) / len(scores)
                        st.markdown(f"**Average Confidence**: {avg_conf:.2f}")
                        
                        # Show individual instances
                        st.markdown("**Individual Detections:**")
                        for i, score in enumerate(scores, 1):
                            st.markdown(f"- Instance {i}: {score:.2f} confidence")
                
                # Show confidence distribution
                st.markdown("### Confidence Distribution")
                all_scores = [det['score'] for det in results['detections']]
                score_ranges = {
                    "Very High (0.9-1.0)": len([s for s in all_scores if s >= 0.9]),
                    "High (0.7-0.9)": len([s for s in all_scores if 0.7 <= s < 0.9]),
                    "Medium (0.5-0.7)": len([s for s in all_scores if 0.5 <= s < 0.7]),
                    "Low (0.0-0.5)": len([s for s in all_scores if s < 0.5])
                }
                
                for range_name, count in score_ranges.items():
                    if count > 0:
                        st.markdown(f"- {range_name}: {count} detections")

if __name__ == "__main__":
    main()
