import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest

# create the sensor_data
np.random.seed(42)

normal_data = np.random.normal(
    loc=50,
    scale=5,
    size=(1000, 3)
)

anomaly_data = np.random.normal(
    loc=90,
    scale=10,
    size=(50, 3)
)

X = np.vstack([normal_data, anomaly_data])
df = pd.DataFrame(
    X,
    columns=["sensor_1", "sensor_2", "sensor_3"]
)

# train model

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(df)

# save model

joblib.dump(
    model,
    "models/anomaly_model.pkl"
)

print(" Model Saved Successfully")