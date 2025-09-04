# Dental Procedure Detection App 🦷

This Streamlit app uses YOLOv8 (via Ultralytics) and OpenCV to detect and visualize dental procedures from uploaded images.

## 🚀 Features
- Upload dental X-ray or procedure images
- Run YOLOv8 inference with bounding box overlays
- View detection results instantly in-browser

## 🧰 Tech Stack
- Python 3.13
- Streamlit 1.39.0
- Ultralytics YOLOv8
- OpenCV
- Pillow

## 📦 Installation (Local)
```bash
git clone https://github.com/yourusername/dental_procedure.git
cd dental_procedure
pip install -r requirements.txt
streamlit run app_dental.py
