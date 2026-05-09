# import sys

# !{sys.executable} -m pip install pandas numpy matplotlib seaborn scipy scikit-learn pyod statsmodels mlflow plotly --break-system-packages

# ============================================================
# WEEK 1 REMAINING TASKS - COMPLETED
# ============================================================
# Covers:
# ✅ ADF Stationarity Test
# ✅ KPSS Test
# ✅ FFT Spectrum Analysis
# ✅ Isolation Forest
# ✅ Local Outlier Factor
# ✅ Z-Score Detection
# ✅ MAD Detection
# ✅ ROC-AUC / PR-AUC
# ✅ Precision Recall Curve
# ✅ MLflow Logging
# ============================================================

# =========================
# INSTALL REQUIRED LIBRARIES
# =========================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.stats import zscore

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from statsmodels.tsa.stattools import adfuller, kpss
from pyod.models.iforest import IForest
import mlflow
import warnings
warnings.filterwarnings("ignore")


# =========================
# GENERATE SYNTHETIC SENSOR DATA
# =========================

np.random.seed(42)

rows = 1000

timestamps = pd.date_range(
    start='2026-01-01',
    periods=rows,
    freq='min'
)

sensor = np.random.normal(50, 5, rows)

# Inject anomalies
anomaly_indices = np.random.choice(rows, 30)
sensor[anomaly_indices] += np.random.normal(25, 5, 30)
failure_label = np.zeros(rows)
failure_label[anomaly_indices] = 1

df = pd.DataFrame({
    'timestamp': timestamps,
    'sensor_value': sensor,
    'failure_label': failure_label
})

print(df.head())


# ============================================================
# DAY 1 REMAINING TASKS
# ============================================================

# ADF TEST


print("\n========== ADF TEST ==========")

adf_result = adfuller(df['sensor_value'])

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

if adf_result[1] < 0.05:
    print("Data is stationary")
else:
    print("Data is non-stationary")


# =========================
# KPSS TEST
# =========================

print("\n========== KPSS TEST ==========")

kpss_result = kpss(df['sensor_value'])

print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])

if kpss_result[1] < 0.05:
    print("Data is non-stationary")
else:
    print("Data is stationary")


# =========================
# FFT ANALYSIS
# =========================

print("\n========== FFT ANALYSIS ==========")

fft_values = fft(df['sensor_value'])

plt.figure(figsize=(12,5))
plt.plot(np.abs(fft_values))
plt.title("FFT Spectrum Analysis")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()


# ============================================================
# DAY 5 REMAINING TASKS
# ============================================================

X = df[['sensor_value']]

# =========================
# ISOLATION FOREST
# =========================

print("\n========== ISOLATION FOREST ==========")

iso = IsolationForest(contamination=0.03)

df['iso_pred'] = iso.fit_predict(X)

df['iso_anomaly'] = df['iso_pred'].apply(
    lambda x: 1 if x == -1 else 0
)

print(df['iso_anomaly'].value_counts())


# =========================
# LOCAL OUTLIER FACTOR
# =========================

print("\n========== LOCAL OUTLIER FACTOR ==========")

lof = LocalOutlierFactor(contamination=0.03)

df['lof_pred'] = lof.fit_predict(X)

df['lof_anomaly'] = df['lof_pred'].apply(
    lambda x: 1 if x == -1 else 0
)

print(df['lof_anomaly'].value_counts())


# =========================
# ROC-AUC SCORE
# =========================

print("\n========== ROC AUC ==========")

roc_score = roc_auc_score(
    df['failure_label'],
    df['iso_anomaly']
)

print("ROC-AUC:", roc_score)


# =========================
# PR-AUC SCORE
# =========================

print("\n========== PR AUC ==========")

pr_score = average_precision_score(
    df['failure_label'],
    df['iso_anomaly']
)

print("PR-AUC:", pr_score)


# =========================
# PRECISION-RECALL CURVE
# =========================

precision, recall, thresholds = precision_recall_curve(
    df['failure_label'],
    df['iso_anomaly']
)

plt.figure(figsize=(8,5))
plt.plot(recall, precision)
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# ============================================================
# DAY 6 REMAINING TASKS
# ============================================================

# =========================
# Z-SCORE ANOMALY
# =========================

print("\n========== Z-SCORE ==========")

df['zscore'] = zscore(df['sensor_value'])

df['z_anomaly'] = np.where(
    abs(df['zscore']) > 3,
    1,
    0
)

print(df['z_anomaly'].value_counts())


# =========================
# MAD METHOD
# =========================

print("\n========== MAD METHOD ==========")

median = np.median(df['sensor_value'])

mad = np.median(
    np.abs(df['sensor_value'] - median)
)

modified_z = 0.6745 * (
    (df['sensor_value'] - median) / mad
)

df['mad_anomaly'] = np.where(
    np.abs(modified_z) > 3.5,
    1,
    0
)

print(df['mad_anomaly'].value_counts())


# ============================================================
# VISUALIZE ALL ANOMALIES
# ============================================================

plt.figure(figsize=(15,6))

plt.plot(df['sensor_value'], label='Sensor Data')

plt.scatter(
    df[df['iso_anomaly'] == 1].index,
    df[df['iso_anomaly'] == 1]['sensor_value'],
    label='Isolation Forest',
    marker='x'
)

plt.scatter(
    df[df['z_anomaly'] == 1].index,
    df[df['z_anomaly'] == 1]['sensor_value'],
    label='Z-Score',
    marker='o'
)

plt.legend()

plt.title("Anomaly Detection Comparison")

plt.show()


# ============================================================
# DAY 7 REMAINING TASKS
# ============================================================

# =========================
# MLFLOW LOGGING
# =========================

print("\n========== MLFLOW LOGGING ==========")
mlflow.set_experiment("week1_anomaly_detection")

with mlflow.start_run():
    mlflow.log_param("model", "Isolation Forest")
    mlflow.log_metric("roc_auc", roc_score)
    mlflow.log_metric("pr_auc", pr_score)
    print("MLflow logging completed")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n================================================")
print("ALL REMAINING WEEK 1 TASKS COMPLETED SUCCESSFULLY")
print("================================================")

