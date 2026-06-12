# SmartShelf AI — Retail Inventory Monitor

A real-time retail shelf monitoring and inventory analytics system using YOLOv8 and OpenCV, featuring a comprehensive Streamlit dashboard.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

## Features
- **Real-Time Detection:** Utilizes the YOLOv8 nano model for rapid inference to identify products on retail shelves.
- **Image Enhancement:** Preprocesses images using Gaussian blur and CLAHE for optimized object detection under varying lighting conditions.
- **Shelf Analytics:** Calculates shelf occupancy percentage and counts total products.
- **Stock Status Alerts:** Automatically determines stock status ("Well Stocked", "Low Stock", "Critical").
- **Empty Zone Detection:** Overlays a 3x3 grid to pinpoint exactly where restocking is required.
- **Interactive Dashboard:** Streamlit UI with adjustable thresholds, Plotly visualizations, and batch processing capabilities.
- **Exportable Reports:** Download annotated images (PNG) and detailed detection reports (CSV).

## Architecture

```mermaid
graph TD
    A[User Image Upload] -->|Streamlit| B(app.py)
    B -->|Preprocess| C(detector.py)
    C -->|CLAHE & Blur| C
    C -->|YOLOv8n Inference| D{Detections}
    D -->|Calculate Occupancy| E(analytics.py)
    D -->|Find Empty Zones| E
    D -->|Stock Status| E
    E -->|Render Overlay| F(utils.py)
    F -->|Return Annotated Image| B
    E -->|Return Report Data| B
    B --> G[Dashboard Metrics & Charts]
    B --> H[Download Reports]
```

## Setup Instructions

1. Clone or download the repository.
2. Ensure you have Python 3.10+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## How to Use

1. Launch the application, which will automatically download the `yolov8n.pt` pretrained weights.
2. Open your web browser to `http://localhost:8501`.
3. Select an **Upload Mode** from the sidebar (Single Image or Batch Mode).
4. Adjust the **Confidence Threshold** and **Low Stock Alert Threshold** to fine-tune the analytics.
5. Upload one or more shelf images.
6. Review the original and annotated images side-by-side.
7. Analyze the dashboard metrics, product distribution bar chart, and shelf occupancy gauge chart.
8. Download the annotated image or CSV report for your records.

## Sample Metrics

*Placeholder results from a typical batch run:*
> **Detected 47 products across 5 shelf images | Avg occupancy: 68% | 2 low-stock alerts triggered**
