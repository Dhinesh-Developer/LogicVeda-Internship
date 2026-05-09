from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

app = FastAPI()

engine = create_engine(
    "postgresql://postgres:arise@localhost:5432/rul_project"
)


@app.get("/")
def home():
    return {
        "message": "Real-Time AI Monitoring API Running"
    }


@app.get("/sensor-data")
def sensor_data():

    query = """
    SELECT *
    FROM sensor_data
    LIMIT 20
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient='records')


@app.get("/anomaly-check")
def anomaly_check():

    query = """
    SELECT *
    FROM sensor_data
    ORDER BY time DESC
    LIMIT 1
    """

    df = pd.read_sql(query, engine)
    latest = df.iloc[0]
    anomaly = "NO"

    if latest["sensor_6"] > 1590:
        anomaly = "YES"

    return {

        "time": str(latest["time"]),
        "unit_id": int(latest["unit_id"]),
        "cycle": int(latest["cycle"]),
        "sensor_1": float(latest["sensor_1"]),
        "sensor_2": float(latest["sensor_2"]),
        "sensor_6": float(latest["sensor_6"]),
        "sensor_7": float(latest["sensor_7"]),
        "sensor_10": float(latest["sensor_10"]),
        "anomaly_detected": anomaly
    }