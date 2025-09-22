# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# -------------------------------
# Helper Functions
# -------------------------------
def build_lstm_model(lookback=60, n_features=2, horizon=5, stack=(64, 64), dropout=0.2):
    model = Sequential()
    for li, units in enumerate(stack):
        return_sequences = (li < len(stack) - 1)
        if li == 0:
            model.add(LSTM(units, return_sequences=return_sequences,
                           input_shape=(lookback, n_features)))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        model.add(Dropout(dropout))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(horizon * 2))  # predict (Close, Volume) × horizon
    model.compile(optimizer="adam", loss="mse")
    return model

def preprocess(df, lookback=60):
    features = ["Close", "Volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    last_seq = np.array([scaled[-lookback:]])  # shape (1, lookback, 2)
    return last_seq, scaler

def forecast(model, seq, scaler, horizon=5):
    y_pred = model.predict(seq).reshape(horizon, 2)  # (5, 2)
    # inverse scale
    pad = np.zeros((horizon, scaler.n_features_in_))
    pad[:, 0] = y_pred[:, 0]  # Close
    pad[:, 1] = y_pred[:, 1]  # Volume
    inv = scaler.inverse_transform(pad)
    pred_df = pd.DataFrame(inv, columns=["Close", "Volume"])
    return pred_df

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="LSTM Stock Forecast", layout="wide")
st.title("📈 Stock Forecast — Close & Volume (LSTM)")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = st.sidebar.slider("Lookback (days)", 30, 120, 60)
horizon = 5

# Load data
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
if df.empty:
    st.error("No data found for this ticker/date range.")
    st.stop()

df = df[["Close", "Volume"]].dropna()

# Preprocess
seq, scaler = preprocess(df, lookback)

# Build + Load trained weights
model = build_lstm_model(lookback=lookback, n_features=2, horizon=horizon, stack=(64, 64))
try:
    model.load_weights("lstmNishe.weights.h5")
    st.success("✅ Loaded trained weights.")
except Exception as e:
    st.warning("⚠️ Could not load weights, using random init.")
    st.text(str(e))

# Run Prediction
if st.sidebar.button("Run Forecast"):
    pred_df = forecast(model, seq, scaler, horizon=horizon)
    pred_df.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                                  periods=horizon, freq="B")

    st.subheader("Predicted Next 5 Days")
    st.write(pred_df)

    # -------------------------------
    # Plot Close
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Historical Close", color="blue")
    ax.plot(pred_df.index, pred_df["Close"], marker="o", color="red", label="Predicted Close")
    ax.set_title(f"{ticker} — Close Price Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Plot Volume
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Volume"], label="Historical Volume", color="blue")
    ax.plot(pred_df.index, pred_df["Volume"], marker="o", color="red", label="Predicted Volume")
    ax.set_title(f"{ticker} — Volume Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Volume")
    ax.legend()
    st.pyplot(fig)
