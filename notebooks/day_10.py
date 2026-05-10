import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

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
y = np.array([0]*1000 + [1]*50)

df = pd.DataFrame(
    X,
    columns=["sensor_1", "sensor_2", "sensor_3"]
)

mlflow.set_experiment("Predictive_Maintenance")
with mlflow.start_run():

    contamination = 0.05
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    model.fit(df)
    preds = model.predict(df)
    preds = np.where(preds == -1, 1, 0)
    accuracy = accuracy_score(y, preds)
    mlflow.log_param(
        "contamination",
        contamination
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        model,
        "isolation_forest_model"
    )
    print("Model Accuracy:", accuracy)

print("MLflow Tracking Completed")


# output
# Model Accuracy: 0.9971428571428571
# MLflow Tracking Completed