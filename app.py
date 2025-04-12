import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.header("Detekcja paneli fotowoltaicznych za pomocą modeli z rodziny YOLO")

uploaded_file = st.file_uploader("Dodaj zdjęcie satelitarne lub ortofotomapy:", # Widżet do uploadu plików
                                 type=["png", "jpg", "jpeg"], accept_multiple_files=False)
selected_model = st.selectbox("Wybierz model:", ("yolov5su", "yolov8s", "yolov10s"), placeholder="Model...") # Widżet do wyboru modelu

if st.button("Uruchom", disabled=uploaded_file is None,): # Przycisk z zabezpieczeniem wyboru pliku
    model = YOLO(f"./models/{selected_model}-solar.pt") # Załadowanie wybranego modelu
    with Image.open(uploaded_file) as image: #Załadowanie obrazu do zmiennej

        st.image(image, caption="Wrzucone zdjęcie") # Wyświetlenie obrazu
        resized_image = np.array(image.resize((640, 640)).convert("RGB")) # Konwersja obrazu do odpowiednich wymiarów dla modelu
        result = model.predict(resized_image, verbose=True) # Predykcja położenia paneli
        annotated_image = result[0].plot() # Obraz z zaznaczonymi panelami za pomocą ramek
        detections = result[0].boxes # Detekcje paneli
        panel_class_id = 0 # identyfikator klasy - 0 oznacza panele fotowoltaiczne
        panel_count = sum(1 for box in detections if box.cls == panel_class_id) # obliczanie ilości detekcji

        st.image(annotated_image, caption=f"Wykryte panele") # Zwrócenie obrazu z zaznaczonymi ramkami
        st.header(f"Znaleziono paneli: {panel_count}") # Zwrócenie liczby paneli
