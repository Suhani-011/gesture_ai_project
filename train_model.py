import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load dataset
data = pd.read_csv("dataset/data.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels (YES, NO → numbers)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20)

# Save model
model.save("model/gesture_model.h5")

print("✅ Model trained and saved!")