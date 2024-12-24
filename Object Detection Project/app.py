from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import base64 

# Fungsi untuk menambahkan latar belakang
def set_background(image_path):
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Fungsi untuk menampilkan GIF di sidebar
def display_animation(gif_path):
    with open(gif_path, "rb") as file:
        gif_data = file.read()
    base64_gif = base64.b64encode(gif_data).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/gif;base64,{base64_gif}" alt="Animation" width="200">
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Function untuk memproses hasil deteksi
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Fungsi utama aplikasi
def main():
    # Set judul aplikasi
    st.title("Live Object Detection Qodrat")
    # Tampilkan animasi di bawah sidebar
    display_animation("Animation - 1734971782012.gif")  # Path ke file GIF
    st.sidebar.title("Live Object Detection Qodrat")
    
    # Atur background
    set_background("background.png")  # Path ke gambar latar belakang
    
    model_path = "yolo11n.pt"  # Path ke model YOLO Anda
    model = load_model(model_path)

    # Inisialisasi status deteksi
    if "run_detection" not in st.session_state:
        st.session_state.run_detection = False

    # Tombol mulai/berhenti deteksi
    st.sidebar.markdown("### MENU")
    if st.sidebar.button("Mulai"):
        st.session_state.run_detection = True

    if st.sidebar.button("Berhenti"):
        st.session_state.run_detection = False

   

    # Jalankan deteksi saat aktif
    if st.session_state.run_detection:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()  # Placeholder untuk video
        st_detection_info = st.empty()  # Placeholder untuk info deteksi

        while st.session_state.run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture image.")
                break

            # Proses deteksi dengan YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
            results = model.predict(frame, imgsz=640) 
            
            # Tampilkan hasil
            frame, detected_objects = display_results(frame, results[0])
            st_frame.image(frame, channels="RGB", use_column_width=True)
            
            # Tampilkan informasi deteksi
            if detected_objects:
                object_counts = Counter(detected_objects)
                detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
            else:
                detection_info = "No objects detected."

            st_detection_info.text(detection_info)

        cap.release()

if __name__ == "__main__":
    main()
