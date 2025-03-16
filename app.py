import gradio as gr
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load trained models & scaler
with open("perceptron_model.pkl", "rb") as f:
    perceptron = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

ann_model = load_model("deep_learning_model.h5")

# Prediction function
def evaluate_employability(appearance, speaking, physical, alertness, confidence, presentation, communication, performance):
    # Convert input to array
    input_data = np.array([[appearance, speaking, physical, alertness, confidence, presentation, communication, performance]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predictions
    perceptron_pred = perceptron.predict(input_scaled)[0]
    ann_pred = ann_model.predict(input_scaled)[0][0]  # ANN outputs probability

    # Convert ANN output to class
    ann_final_pred = 1 if ann_pred >= 0.5 else 0  # Threshold at 0.5

    # Map predictions to labels
    perceptron_result = "Employable " if perceptron_pred == 1 else "LessEmployable "
    ann_result = "Employable " if ann_final_pred == 1 else "LessEmployable "
    
    return f"Perceptron Prediction: {perceptron_result}\nDeep Learning Prediction: {ann_result} (Confidence: {ann_pred:.2f})"

# Gradio UI
iface = gr.Interface(
    fn=evaluate_employability,
    inputs=[
        gr.Slider(1, 5, step=1, label="General Appearance"),
        gr.Slider(1, 5, step=1, label="Manner of Speaking"),
        gr.Slider(1, 5, step=1, label="Physical Condition"),
        gr.Slider(1, 5, step=1, label="Mental Alertness"),
        gr.Slider(1, 5, step=1, label="Self-Confidence"),
        gr.Slider(1, 5, step=1, label="Ability to Present Ideas"),
        gr.Slider(1, 5, step=1, label="Communication Skills"),
        gr.Slider(1, 5, step=1, label="Student Performance Rating")
    ],
    outputs="text",
    title=" AI-Powered Employability Evaluation Tool",
    description="Enter ratings (1-5) for different employability factors, and the AI model will predict whether the candidate is Employable or LessEmployable."
)

# Launch the Gradio app
iface.launch()
