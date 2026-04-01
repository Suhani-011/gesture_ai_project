import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import keyboard
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("model/gesture_model.h5")

with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)

# ---------------- CONTROL ----------------
last_action_time = 0
cooldown = 2.5

prev_label = ""
stable_count = 0

popup_text = ""
popup_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = "Detecting"
    hand_type = ""
    cx, cy = 0, 0

    if results.multi_hand_landmarks:
        for i, handLms in enumerate(results.multi_hand_landmarks):

            if results.multi_handedness:
                hand_type = results.multi_handedness[i].classification[0].label

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            # -------- COORDINATES --------
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

            # -------- DATA --------
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

            # -------- STABILITY --------
            if label == prev_label:
                stable_count += 1
            else:
                stable_count = 0

            prev_label = label
            current_time = time.time()

            # -------- ACTION --------
            if stable_count > 12 and (current_time - last_action_time > cooldown):

                if label in ["PAUSE", "PLAY"]:
                    keyboard.send("play/pause media")
                    popup_text = "PLAY / PAUSE"

                elif label == "VOLUP":
                    for _ in range(5):
                        keyboard.send("volume up")
                        time.sleep(0.03)
                    popup_text = "VOLUME UP"

                elif label == "VOLDOWN":
                    for _ in range(5):
                        keyboard.send("volume down")
                        time.sleep(0.03)
                    popup_text = "VOLUME DOWN"

                elif label == "NEXT":
                    keyboard.send("next track")
                    popup_text = "NEXT"

                elif label == "PREV":
                    keyboard.send("previous track")
                    popup_text = "PREVIOUS"

                popup_time = current_time
                last_action_time = current_time
                stable_count = 0

    # -------- MINIMAL HEADER --------
    cv2.putText(frame, "Gesture Control",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 0, 0), 2)

    # -------- INFO TEXT (NO BOXES) --------
    cv2.putText(frame, f"{label}",
                (10, 70),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 0, 0), 2)

    cv2.putText(frame, f"{hand_type}",
                (10, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (50, 50, 50), 1)

    cv2.putText(frame, f"({cx}, {cy})",
                (10, 125),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (80, 80, 80), 1)

    # -------- CLEAN POPUP --------
    if time.time() - popup_time < 1.2:
        cv2.putText(frame, popup_text,
                    (frame.shape[1]//2 - 100, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0), 3)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()