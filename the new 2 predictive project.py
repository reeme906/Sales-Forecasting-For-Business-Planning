
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


df = pd.read_csv("train.csv")
df = df.head(1000)


df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

daily = df.groupby("Date", as_index=False)["Weekly_Sales"].sum()

data = daily.set_index("Date").sort_index()

data = data.asfreq('W-FRI')

data["Weekly_Sales"] = data["Weekly_Sales"].ffill().bfill()


plt.figure(figsize=(10,4))
plt.plot(data.index, data["Weekly_Sales"], marker="o")
plt.title("Weekly Sales (First 1000 Rows) - Line Chart")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.hist(data["Weekly_Sales"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribution of Weekly Sales (First 1000 Rows)")
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(data["Weekly_Sales"])
plt.title("Weekly Sales - Boxplot (First 1000 Rows)")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.scatter(data.index, data["Weekly_Sales"], color="red")
plt.title("Weekly Sales - Scatter Plot (First 1000 Rows)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.show()


train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

print("Train size:", len(train))
print("Test size :", len(test))


model = ARIMA(train["Weekly_Sales"], order=(1,1,1))
model_fit = model.fit()

arima_pred = model_fit.forecast(steps=len(test))
arima_pred = pd.Series(arima_pred, index=test.index)


plt.figure(figsize=(10,4))
plt.plot(train.index, train["Weekly_Sales"], label="Train Data")
plt.plot(test.index, test["Weekly_Sales"], label="Actual Test Data", marker="o")
plt.plot(arima_pred.index, arima_pred, label="ARIMA Predicted", marker="x")
plt.title("Actual vs Predicted Weekly Sales (ARIMA, First 1000 Rows)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.legend()
plt.show()


arima_mae = mean_absolute_error(test["Weekly_Sales"], arima_pred)
arima_mse = mean_squared_error(test["Weekly_Sales"], arima_pred)
arima_rmse = math.sqrt(arima_mse)

print("\nARIMA Performance:")
print("MAE :", arima_mae)
print("MSE :", arima_mse)
print("RMSE:", arima_rmse)



values = data["Weekly_Sales"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

window = 5
X = []
y = []

for i in range(window, len(scaled_values)):
    X.append(scaled_values[i-window:i, 0])
    y.append(scaled_values[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size_lstm = int(len(X) * 0.8)
X_train, X_test = X[:train_size_lstm], X[train_size_lstm:]
y_train, y_test = y[:train_size_lstm], y[train_size_lstm:]

model_lstm = Sequential()
model_lstm.add(LSTM(32, activation='tanh', return_sequences=False, input_shape=(window, 1)))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.fit(X_train, y_train, epochs=30, batch_size=8, verbose=1)

lstm_scaled_preds = model_lstm.predict(X_test)
lstm_preds = scaler.inverse_transform(lstm_scaled_preds)

lstm_pred_series = pd.Series(
    lstm_preds.flatten(),
    index=data.index[-len(lstm_preds):]
)


plt.figure(figsize=(10,4))
plt.plot(data.index, data["Weekly_Sales"], label="Actual Data")
plt.plot(lstm_pred_series.index, lstm_pred_series.values, label="LSTM Predictions")
plt.title("LSTM Sales Forecasting (First 1000 Rows)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.legend()
plt.show()


lstm_mae = mean_absolute_error(data["Weekly_Sales"].iloc[-len(lstm_preds):], lstm_pred_series)
lstm_mse = mean_squared_error(data["Weekly_Sales"].iloc[-len(lstm_preds):], lstm_pred_series)
lstm_rmse = math.sqrt(lstm_mse)

print("\nLSTM Performance:")
print("MAE :", lstm_mae)
print("MSE :", lstm_mse)
print("RMSE:", lstm_rmse)
