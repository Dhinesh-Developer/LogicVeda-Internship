# import sys
# !{sys.executable} -m pip install plotly scikit-learn joblib --break-system-packages

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="AI Monitoring Dashboard",
    layout="wide"
)

st.title("🚀 Real-Time AI Monitoring Dashboard")

# DATABASE CONNECTION

engine = create_engine(
    "postgresql://postgres:arise@localhost:5432/kafka_db"
)

# LOAD DATA
query = "SELECT * FROM users"
df = pd.read_sql(query, engine)
st.subheader("📊 Incoming Data")
st.dataframe(df)


# CREATE NUMERIC FEATURE
df['name_length'] = df['name'].apply(len)

# TRAIN MODEL
model = IsolationForest(
    contamination=0.1,
    random_state=42
)
df['anomaly'] = model.fit_predict(df[['name_length']])



df['anomaly'] = df['anomaly'].map({
    1: "Normal",
    -1: "Anomaly"
})


# METRICS
col1, col2 = st.columns(2)
col1.metric("Total Records", len(df))
col2.metric(
    "Total Anomalies",
    len(df[df['anomaly'] == "Anomaly"])
)


fig = px.scatter(
    df,
    x=df.index,
    y='name_length',
    color='anomaly',
    title="Real-Time Anomaly Detection"
)

st.plotly_chart(fig, use_container_width=True)

# SHOW ANOMALIES
st.subheader("⚠️ Detected Anomalies")
anomalies = df[df['anomaly'] == "Anomaly"]
st.dataframe(anomalies)






