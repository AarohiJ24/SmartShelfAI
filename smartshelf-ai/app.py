import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

from detector import SmartShelfDetector
from analytics import product_count, shelf_occupancy, stock_status, empty_zone_detection, generate_report
from utils import draw_grid_overlay, draw_empty_zones, resize_for_display

st.set_page_config(page_title="SmartShelf AI — Retail Inventory Monitor", layout="wide")

# Initialize detector only once
@st.cache_resource
def load_detector():
    return SmartShelfDetector("yolov8n.pt")

detector = load_detector()

st.title("SmartShelf AI — Retail Inventory Monitor")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
low_stock_threshold = st.sidebar.slider("Low Stock Alert Threshold (%)", min_value=10, max_value=80, value=30, step=5)

upload_mode = st.sidebar.radio("Upload Mode", ["Single Image", "Batch Mode"])

uploaded_files = st.sidebar.file_uploader("Upload Shelf Image(s)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=(upload_mode == "Batch Mode"))

def process_single_image(image_file):
    try:
        # Reset file pointer just in case
        image_file.seek(0)
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        if image_bgr is None:
            st.error("Error decoding image.")
            return None
        # Convert BGR to RGB
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

    # Detect
    annotated_image, detections = detector.detect(image_np, conf_threshold=conf_threshold)
    
    h, w = image_np.shape[:2]
    
    # Analytics
    count = product_count(detections)
    occupancy = shelf_occupancy(detections, w, h)
    status = stock_status(occupancy, low_stock_threshold)
    empty_zones = empty_zone_detection(detections, w, h)
    
    # Apply Utils drawings
    annotated_image = draw_grid_overlay(annotated_image)
    annotated_image = draw_empty_zones(annotated_image, empty_zones)
    
    report_df = generate_report(detections, w, h)
    report_df.insert(0, "Image Name", image_file.name)
    
    return {
        "image_name": image_file.name,
        "original_image": image_np,
        "annotated_image": annotated_image,
        "count": count,
        "occupancy": occupancy,
        "status": status,
        "empty_zones": empty_zones,
        "report_df": report_df
    }

if uploaded_files:
    if upload_mode == "Single Image":
        # Handle case where user switched from Batch to Single but uploaded_files is a list
        file = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
        
        with st.spinner("Processing image..."):
            result = process_single_image(file)
        
        if result:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(result["original_image"], channels="RGB", use_column_width=True)
            with col2:
                st.subheader("Annotated Image")
                st.image(result["annotated_image"], channels="RGB", use_column_width=True)
                
            st.markdown("---")
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Products Detected", result["count"])
            m2.metric("Shelf Occupancy %", f"{result['occupancy']}%")
            m3.metric("Empty Zones Count", len(result["empty_zones"]))
            
            # Color code status
            status_color = "green" if result["status"] == "Well Stocked" else "orange" if result["status"] == "Low Stock" else "red"
            m4.markdown(f"### Stock Status\n<span style='color:{status_color}; font-weight:bold; font-size:20px'>{result['status']}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Plotly bar chart
                if not result["report_df"].empty:
                    class_counts = result["report_df"]["Product Class"].value_counts().reset_index()
                    class_counts.columns = ["Product Class", "Count"]
                    fig_bar = px.bar(class_counts, x="Product Class", y="Count", title="Product Class Distribution")
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No products detected for distribution chart.")
                    
            with chart_col2:
                # Plotly gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result["occupancy"],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Shelf Occupancy %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, low_stock_threshold], 'color': "red"},
                            {'range': [low_stock_threshold, 50], 'color': "yellow"},
                            {'range': [50, 100], 'color': "green"}],
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            st.subheader("Detection Report")
            st.dataframe(result["report_df"])
            
            # Downloads
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                # Image download
                img_pil = Image.fromarray(result["annotated_image"])
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button("Download Annotated Image (PNG)", data=buf.getvalue(), file_name=f"annotated_{result['image_name']}.png", mime="image/png")
                
            with dl_col2:
                # CSV download
                csv = result["report_df"].to_csv(index=False)
                st.download_button("Download Report (CSV)", data=csv, file_name=f"report_{result['image_name']}.csv", mime="text/csv")
                
    else: # Batch Mode
        st.header("Batch Processing Results")
        all_reports = []
        summary_data = []
        
        # Ensure uploaded_files is a list
        files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(files_to_process):
            result = process_single_image(file)
            if result:
                all_reports.append(result["report_df"])
                summary_data.append({
                    "Image Name": result["image_name"],
                    "Total Detected": result["count"],
                    "Occupancy %": result["occupancy"],
                    "Empty Zones": len(result["empty_zones"]),
                    "Status": result["status"]
                })
            progress_bar.progress((idx + 1) / len(files_to_process))
            
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.subheader("Aggregate Summary")
            
            # Highlight low stock
            def highlight_status(val):
                color = 'red' if val == 'Critical - Restock Needed' else 'orange' if val == 'Low Stock' else 'green'
                return f'color: {color}'
                
            st.dataframe(summary_df.style.map(highlight_status, subset=['Status']))
            
            if all_reports:
                full_report_df = pd.concat(all_reports, ignore_index=True)
                st.subheader("Full Combined Detection Report")
                st.dataframe(full_report_df)
                
                # CSV download
                csv = full_report_df.to_csv(index=False)
                st.download_button("Download Full Report (CSV)", data=csv, file_name="batch_report.csv", mime="text/csv")
else:
    st.info("Please upload an image to get started.")
