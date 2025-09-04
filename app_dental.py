import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import io

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI setup
st.set_page_config(page_title="Dental Procedure Detection", layout="centered")
st.title("🦷 Dental Procedure Detection App")
st.write("Upload a dental image to detect procedures using YOLOv8.")

# Upload image
uploaded_file = st.file_uploader("📁 Select an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(pil_img):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return img_np

def draw_boxes(image_np, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} ({conf:.2f})"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_np

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    image_np = preprocess_image(image)

    if st.button("🔍 Run Detection"):
        results = model.predict(image_np, save=False)
        image_np = draw_boxes(image_np, results)
        detected_image = Image.fromarray(image_np)
        st.image(detected_image, caption="✅ Detection Result", use_column_width=True)

        # Convert image to bytes for browser download
        img_bytes = io.BytesIO()
        detected_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        st.download_button(
            label="💾 Download Detected Image",
            data=img_bytes,
            file_name="detected_image.png",
            mime="image/png"
        )

# Footer
st.markdown("---")
st.caption("Developed by Shahab | Powered by DENTAL_PROCEDURE_DETECTION 🦷")
