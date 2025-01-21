import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

st.header("Znajdź Panele")

uploaded_file = st.file_uploader("Dodaj zdjęcie satelitarne lub ortofotomapy:", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
selected_model = st.selectbox("Wybierz model:", ("yolov5su", "yolov8s", "yolov10s"), placeholder="Model...")

if st.button("Uruchom", disabled=uploaded_file is None,):
    st.write("Ładowanie modelu")
    model = YOLO(f"./models/{selected_model}-solar.pt")
    with Image.open(uploaded_file) as image:

        st.image(image, caption="Wrzucone zdjęcie")
        resized_image = np.array(image.resize((640, 640)).convert("RGB"))
        result = model.predict(resized_image, verbose=True)
        annotated_image = result[0].plot()
        detections = result[0].boxes
        panel_class_id = 0
        panel_count = sum(1 for box in detections if box.cls == panel_class_id)

        st.image(annotated_image, caption=f"Wykryte panele")
        st.header(f"Znaleziono paneli: {panel_count}")
