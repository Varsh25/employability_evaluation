import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
file_path = "employability.xlsx"  # Change path if needed
df = pd.read_excel(file_path)

# Drop unnecessary columns (e.g., "Name of Student")
df = df.drop(columns=["Name of Student"])

# Encode target variable (Employable → 1, LessEmployable → 0)
label_encoder = LabelEncoder()
df["CLASS"] = label_encoder.fit_transform(df["CLASS"])

# Split features and target
X = df.drop(columns=["CLASS"])
y = df["CLASS"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Perceptron Model
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X_train, y_train)

# Train Deep Learning Model (ANN)
ann_model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann_model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Save models and scaler
pickle.dump(perceptron, open("perceptron_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
ann_model.save("deep_learning_model.h5")

print("Models and scaler saved successfully!")

# Download files for Hugging Face deployment
from google.colab import files
files.download("perceptron_model.pkl")
files.download("scaler.pkl")
files.download("deep_learning_model.h5")
