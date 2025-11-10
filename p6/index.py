import streamlit as st
import cv2
import requests
import time

st.title("Real-Time Emotion Detection Using ViT Architecture")

# Start/Stop Button
run = st.checkbox("Start Camera")

# Create image display slot
frame_window = st.empty()

cap = None

while run:
    if cap is None:
        cap = cv2.VideoCapture(0)   # open only once

    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not detected.")
        break

    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)

    # Send to FastAPI
    try:
        r = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        )
        emotion = r.json()["emotion"]
    except:
        emotion = "..."

    # Draw prediction
    cv2.putText(frame, emotion, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    # Display frame (updates in-place, no duplication!)
    frame_window.image(frame, channels="BGR")

    time.sleep(0.01)

# Release camera when checkbox is unticked
if cap:
    cap.release()
