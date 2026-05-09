import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Monitoring Dashboard",
    layout="wide"
)

st.title("🚀 Real-Time AI Monitoring Dashboard")

data = requests.get(
    "http://127.0.0.1:8000/sensor-data"
).json()

df = pd.DataFrame(data)
st.subheader("📊 Live Sensor Data")
st.dataframe(df)
col1, col2, col3 = st.columns(3)
col1.metric(
    "Total Records",
    len(df)
)

col2.metric(
    "Average Sensor 6",
    round(df["sensor_6"].mean(), 2)
)

col3.metric(
    "Average Sensor 7",
    round(df["sensor_7"].mean(), 2)
)

st.subheader("📈 Sensor 6 Trend")

fig = px.line(
    df,
    x="cycle",
    y="sensor_6",
    title="Sensor 6 Values"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("⚠️ Latest Anomaly Detection")

anomaly = requests.get(
    "http://127.0.0.1:8000/anomaly-check"
).json()

st.json(anomaly)