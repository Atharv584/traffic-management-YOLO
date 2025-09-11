import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Load trained YOLO model
model = YOLO("best.pt")

st.set_page_config(page_title="Traffic Lane Analyzer", layout="wide")
st.title("Smart Traffic Management")

# Let the user choose the number of lanes
num_lanes = st.slider("Select the number of lanes:", 2, 8, 4)
lanes = [f"Lane {i+1}" for i in range(num_lanes)]

# Store uploaded files
uploaded_files = {}

# Upload section with better alignment, dynamically creating columns
st.header("Upload Images for All Lanes")
cols = st.columns(num_lanes)
for i, lane in enumerate(lanes):
    with cols[i]:
        uploaded_files[lane] = st.file_uploader(
            f"Upload image for {lane}", type=["jpg", "jpeg", "png"], key=lane
        )

# Base time allocation rules (sec) for traffic levels
base_time_allocation = {
    "Traffic Jam": 100,
    "High": 75,
    "Medium": 50,
    "Low": 25,
    "Empty": 10
}

# Submit button
if st.button("Submit"):
    all_uploaded = all(uploaded_files.values())
    
    if not all_uploaded:
        st.error("Please upload images for all selected lanes before submitting.")
    else:
        results_dict = {}
        lane_times = {}

        # Prepare row for images, with columns based on the number of lanes
        image_cols = st.columns(num_lanes)

        for i, (lane, file) in enumerate(uploaded_files.items()):
            image = Image.open(file)

            # Run YOLO prediction
            results = model.predict(np.array(image), imgsz=224, verbose=False)

            # Extract prediction (classification model case)
            pred_id = results[0].probs.top1
            pred_name = results[0].names[pred_id]
            pred_conf = results[0].probs.top1conf.item()

            # Confidence-based adjustment
            base_time = base_time_allocation.get(pred_name, 20)
            adjusted_time = int(max(5, base_time * pred_conf))  # min 5 sec

            results_dict[lane] = (pred_name, pred_conf)
            lane_times[lane] = adjusted_time

            # Show images in one row
            with image_cols[i]:
                st.image(image, caption=f"{lane}\n{pred_name} ({pred_conf:.2f})", width=200)

        # Show results in table
        st.header("Results with Lane Timings")
        results_df = pd.DataFrame([
            {
                "Lane": lane,
                "Traffic Level": pred_name,
                "Confidence": f"{pred_conf:.2f}",
                "Green Light Time (sec)": lane_times[lane]
            }
            for lane, (pred_name, pred_conf) in results_dict.items()
        ])

        # Sort the DataFrame by 'Green Light Time (sec)' in descending order
        sorted_df = results_df.sort_values(by="Green Light Time (sec)", ascending=False)
        
        # Display the sorted DataFrame
        st.table(sorted_df.set_index('Lane'))