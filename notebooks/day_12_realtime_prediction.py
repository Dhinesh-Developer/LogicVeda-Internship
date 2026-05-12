from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np

model = joblib.load(
    "models/anomaly_model.pkl"
)

app = FastAPI()

class SensorData(BaseModel):

    sensor_1: float
    sensor_2: float
    sensor_3: float

@app.get("/")
def home():

    return {
        "message": "Real-Time Prediction API Running"
    }


@app.post("/predict")
def predict(data: SensorData):

    sample = np.array([
        [
            data.sensor_1,
            data.sensor_2,
            data.sensor_3
        ]
    ])

    prediction = model.predict(sample)[0]

    result = "NORMAL"

    if prediction == -1:
        result = "ANOMALY"

    return {

        "sensor_data": {

            "sensor_1": data.sensor_1,
            "sensor_2": data.sensor_2,
            "sensor_3": data.sensor_3
        },

        "prediction": result
    }