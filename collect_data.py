import cv2
import mediapipe as mp
import csv
import os

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

file = open("dataset/data.csv", mode='a', newline='')
writer = csv.writer(file)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter gesture name (e.g. YES, NO, OK): ")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            data = []

            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)

            data.append(label)
            writer.writerow(data)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", img)

    # Press 's' to save, ESC to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()