import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load trained YOLO model
model = YOLO("best.pt")

st.title("🚦 Traffic Lane Analyzer")

# Define 4 lanes
lanes = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

# Store uploaded files
uploaded_files = {}

st.header("Upload Images for All Lanes")
for lane in lanes:
    uploaded_files[lane] = st.file_uploader(
        f"Upload image for {lane}", type=["jpg", "jpeg", "png"], key=lane
    )

# Submit button
if st.button("Submit"):
    all_uploaded = all(uploaded_files.values())
    
    if not all_uploaded:
        st.error("⚠️ Please upload images for all 4 lanes before submitting.")
    else:
        st.success("✅ Processing images...")

        results_dict = {}
        for lane, file in uploaded_files.items():
            image = Image.open(file)
            st.image(image, caption=f"{lane} Image", use_container_width=True)

            # Run YOLO prediction
            results = model.predict(np.array(image), imgsz=224, verbose=False)

            # Extract prediction
            pred_id = results[0].probs.top1
            pred_name = results[0].names[pred_id]
            pred_conf = results[0].probs.top1conf.item()

            results_dict[lane] = (pred_name, pred_conf)

        # Show results together
        st.header("📊 Prediction Results")
        for lane, (pred_name, pred_conf) in results_dict.items():
            st.write(f"**{lane}** → {pred_name} (Confidence: {pred_conf:.2f})")
