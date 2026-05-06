import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# PAGE CONFIG
st.set_page_config(page_title="Kafka Dashboard", layout="wide")
st.title("📡 Real-Time Kafka Dashboard")


# DATABASE CONNECTION
@st.cache_resource
def get_connection():
    engine = create_engine("postgresql://postgres:arise@localhost:5432/kafka_db")
    return engine

engine = get_connection()

# FETCH DATA
@st.cache_data(ttl=5)
def load_data():
    query = "SELECT * FROM users ORDER BY id DESC LIMIT 100"
    df = pd.read_sql(query, engine)
    return df

df = load_data()


# METRICS
st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Unique Cities", df['city'].nunique())
col3.metric("Latest User", df['name'].iloc[0] if len(df) > 0 else "N/A")

# TABLE
st.subheader("Latest Records")
st.dataframe(df, use_container_width=True)

# CHARTS
st.subheader("Analytics")

col1, col2 = st.columns(2)

with col1:
    st.write("Users by City")
    st.bar_chart(df['city'].value_counts())

with col2:
    st.write("⏱ Records Over Time")
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp')
            st.line_chart(df.set_index('timestamp'))
        except:
            st.write("Timestamp format not supported")


# AUTO REFRESH BUTTON
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()
