from fastapi import FastAPI
from pydantic import BaseModel

from sqlalchemy import create_engine
from sqlalchemy import text

import numpy as np
import joblib

model = joblib.load(
    "models/anomaly_model.pkl"
)


engine = create_engine(
    "postgresql://postgres:arise@localhost:5432/rul_project"
)

app = FastAPI()

class SensorData(BaseModel):

    sensor_1: float
    sensor_2: float
    sensor_3: float

@app.get("/")
def home():

    return {
        "message": "Prediction Logging API Running"
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
    query = text("""

        INSERT INTO prediction_logs (

            sensor_1,
            sensor_2,
            sensor_3,
            prediction

        )

        VALUES (

            :sensor_1,
            :sensor_2,
            :sensor_3,
            :prediction
        )

    """)

    with engine.connect() as conn:

        conn.execute(

            query,

            {

                "sensor_1": data.sensor_1,
                "sensor_2": data.sensor_2,
                "sensor_3": data.sensor_3,
                "prediction": result
            }

        )

        conn.commit()

    return {

        "sensor_data": {

            "sensor_1": data.sensor_1,
            "sensor_2": data.sensor_2,
            "sensor_3": data.sensor_3
        },

        "prediction": result,

        "status": "Saved to Database"
    }