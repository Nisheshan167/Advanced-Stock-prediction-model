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
# Model architecture (MUST MATCH COLAB)
# -------------------------------
def build_trained_model(lookback=60, n_features=2, horizon=5):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(horizon * 2)  # Close + Volume for each day
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# Load trained model weights
# -------------------------------
@st.cache_resource
def load_model(weights_path="lstm.weights.h5"):
    model = build_trained_model(lookback=60, n_features=2, horizon=5)
    try:
        model.load_weights(weights_path)
        st.success("✅ Loaded trained weights successfully.")
    except Exception as e:
        st.warning(f"⚠️ Could not load weights, using random init.\n\nError: {e}")
    return model

model = load_model("lstm.weights.h5")   # change to lstmN.weights.h5 if needed

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess(df, lookback=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close", "Volume"]])
    last_seq = scaled[-lookback:]  # last window
    return scaler, np.array([last_seq])

def forecast_next_days(model, scaler, seq, horizon=5):
    preds = model.predict(seq, verbose=0).reshape(horizon, 2)
    inv = scaler.inverse_transform(preds)
    return pd.DataFrame(inv, columns=["Pred_Close", "Pred_Volume"])

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📈 LSTM Stock Forecast", layout="wide")
st.title("📈 Stock Price & Volume Prediction (LSTM)")

# Sidebar Inputs
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = 60
horizon = 5

if st.sidebar.button("Run Forecast"):
    # Download Data
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df = df[["Close", "Volume"]].dropna()

    # Preprocess
    scaler, seq = preprocess(df, lookback=lookback)

    # Forecast
    pred_df = forecast_next_days(model, scaler, seq, horizon=horizon)
    pred_df.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="B")

    # Display Table
    st.subheader("📊 Predicted Next 5 Days (Close & Volume)")
    st.write(pred_df)

    # Plot Close
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Close Plot
    axes[0].plot(df.index, df["Close"], label="Historical Close", color="blue")
    axes[0].plot(pred_df.index, pred_df["Pred_Close"], marker="o", color="red", label="Predicted Close")
    axes[0].set_title(f"{ticker} — Close Price Forecast")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()

    # Volume Plot
    axes[1].plot(df.index, df["Volume"], label="Historical Volume", color="blue")
    axes[1].plot(pred_df.index, pred_df["Pred_Volume"], marker="o", color="red", label="Predicted Volume")
    axes[1].set_title(f"{ticker} — Volume Forecast")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Volume")
    axes[1].legend()

    st.pyplot(fig)
