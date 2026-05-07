
# DAY 6 - LSTM ANOMALY DETECTION
# Kafka + PostgreSQL + PyTorch
# ================================

# import sys

# !{sys.executable} -m pip install pandas numpy matplotlib sqlalchemy psycopg2-binary scikit-learn torch --break-system-packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler


engine = create_engine(
    "postgresql://postgres:arise@localhost:5432/kafka_db"
)
query = "SELECT * FROM users"
df = pd.read_sql(query, engine)
print(df.head())


np.random.seed(42)
df['sensor_value'] = np.random.normal(
    loc=50,
    scale=5,
    size=len(df)
)

df.loc[10:15, 'sensor_value'] = 90
df.loc[40:45, 'sensor_value'] = 100
print(df.head())


plt.figure(figsize=(15,5))
plt.plot(df['sensor_value'])
plt.title("Sensor Data")
plt.xlabel("Index")
plt.ylabel("Sensor Value")
plt.show()


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(
    df[['sensor_value']]
)

sequence_length = 10
X = []
for i in range(len(data_scaled) - sequence_length):
    X.append(
        data_scaled[i:i+sequence_length]
    )
X = np.array(X)

print("Sequence Shape:", X.shape)



X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)


class LSTMAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=16,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=16,
            hidden_size=1,
            batch_first=True
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


model = LSTMAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 20
for epoch in range(epochs):
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


with torch.no_grad():
    predictions = model(X_tensor)

mse = np.mean(
    (predictions.numpy() - X_tensor.numpy())**2,
    axis=(1,2)
)

threshold = np.percentile(mse, 95)
anomalies = mse > threshold
print("\nThreshold:", threshold)
print("\nAnomalies Found:\n")
print(anomalies)



plt.figure(figsize=(15,5))

plt.plot(
    mse,
    label='Reconstruction Error'
)

plt.axhline(
    threshold,
    color='red',
    linestyle='--',
    label='Threshold'
)

plt.title("LSTM Autoencoder Anomaly Detection")
plt.xlabel("Sequence")
plt.ylabel("Error")
plt.legend()
plt.show()



anomaly_indices = np.where(anomalies == True)
print("\nAnomaly Indexes:\n")
print(anomaly_indices)

