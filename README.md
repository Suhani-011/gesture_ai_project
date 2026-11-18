
# Hand Gesture Recognition & Media Control

A Python-based hand gesture recognition project that uses a webcam to detect hand landmarks, classify trained gestures, and control media playback and volume using keyboard media commands.

## Features

- Real-time hand detection using MediaPipe Hands.
- Webcam processing using OpenCV.
- Collection of hand landmark data for gesture training.
- Neural network-based gesture classification using TensorFlow/Keras.
- Gesture-based media controls:
  - PLAY / PAUSE – Play or pause media.
  - VOLUP – Increase volume.
  - VOLDOWN – Decrease volume.
  - NEXT – Play the next track.
  - PREV – Play the previous track.
- Confidence threshold and gesture stability checks to improve recognition.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Keyboard

## Project Structure

```text
gesture_ai_project/
│
├── dataset/
│   └── data.csv
│
├── model/
│   ├── gesture_model.h5
│   └── label_encoder.pkl
│
├── collect_data.py
├── hand_detection.py
├── train_model.py
├── predict.py
└── README.md
```

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Suhani-011/gesture_ai_project.git
cd gesture_ai_project
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn tensorflow keyboard
```

**4. Create the model folder**

```bash
mkdir model
```

## How to Run the Project

### Step 1: Hand Detection

Run the following command to check real-time hand detection:

```bash
python hand_detection.py
```

### Step 2: Collect Gesture Data

Run:

```bash
python collect_data.py
```

Collect hand gesture samples using your webcam. Save samples for each gesture you want the model to recognize.

Supported gesture labels:

- PLAY
- PAUSE
- VOLUP
- VOLDOWN
- NEXT
- PREV

The collected hand landmark data and labels are stored in `dataset/data.csv`.

### Step 3: Train the Model

After collecting sufficient gesture samples, run:

```bash
python train_model.py
```

The training script processes the collected data and trains a neural network to recognize different hand gestures.

The trained model and label encoder are saved in the model folder.

### Step 4: Run Gesture Recognition

Execute:

```bash
python predict.py
```

Show your trained hand gestures in front of the webcam to control media playback and volume.

Press **Esc** to exit the application.

## How It Works

1. **Hand Detection:** MediaPipe detects hand landmarks through the webcam.
2. **Data Collection:** The system collects hand landmark coordinates and assigns gesture labels.
3. **Model Training:** TensorFlow/Keras trains a neural network using the collected dataset.
4. **Gesture Prediction:** The trained model identifies gestures in real time.
5. **Media Control:** Recognized gestures trigger corresponding media commands through keyboard controls.

## Requirements

- Python installed on your system.
- Working webcam.
- Required Python libraries installed.
- Training dataset collected before model training.
- Trained model and label encoder available before running prediction.

## Author

**Suhani Jain**

GitHub: https://github.com/Suhani-011

## Project Repository

https://github.com/Suhani-011/gesture_ai_project
