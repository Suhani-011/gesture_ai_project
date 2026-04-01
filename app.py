import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Gesture AI", layout="centered")

st.markdown("<h1 style='text-align:center;'>🤖 Gesture Recognition</h1>", unsafe_allow_html=True)

model = load_model("model/gesture_model.h5")

with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

run = st.toggle("🎥 Start Camera")

frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = ""
    hand_type = ""

    if results.multi_hand_landmarks:
        for i, handLms in enumerate(results.multi_hand_landmarks):

            if results.multi_handedness:
                hand_type = results.multi_handedness[i].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                handLms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(220, 220, 220), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(100, 100, 100), thickness=2)
            )

            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in handLms.landmark]
            y_list = [int(lm.y * h) for lm in handLms.landmark]

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            cv2.rectangle(frame, (xmin-25, ymin-25), (xmax+25, ymax+25), (80,80,80), 2)

            data = []
            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)

            data = np.array(data).reshape(1, -1)

            pred = model.predict(data, verbose=0)
            confidence = np.max(pred)

            if confidence > 0.6:
                class_id = np.argmax(pred)
                label = encoder.inverse_transform([class_id])[0]

                # OK FIX
                thumb_tip = handLms.landmark[4]
                index_tip = handLms.landmark[8]

                dist = ((thumb_tip.x - index_tip.x)**2 +
                        (thumb_tip.y - index_tip.y)**2)**0.5

                if dist < 0.05:
                    label = "OK"
            else:
                label = "Detecting..."

            cv2.rectangle(frame, (xmin-25, ymin-80), (xmax+25, ymin-25), (20,20,20), -1)

            text = f"{hand_type} | {label}"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = xmin + (xmax-xmin)//2 - size[0]//2

            cv2.putText(frame, text, (text_x, ymin-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()