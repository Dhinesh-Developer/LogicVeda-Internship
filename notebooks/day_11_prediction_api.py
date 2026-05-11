from fastapi import FastAPI
import joblib
import numpy as np


model = joblib.load(
    "models/anomaly_model.pkl"
)
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "AI Prediction API Running"
    }

@app.get("/predict")
def predict():

    sample_data = np.array([
        [55, 52, 50]
    ])

    prediction = model.predict(sample_data)[0]
    result = "NORMAL"

    if prediction == -1:
        result = "ANOMALY"

    return {
        "prediction": result
    }