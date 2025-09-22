# app.py (Close-only version)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime

# -----------------------------
# Build the SAME architecture as in Colab (Close only)
# -----------------------------
def build_model(lookback=60, horizon=5):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),  # 1 feature
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(horizon)  # only Close (5 steps ahead)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

@st.cache_resource
def load_model(weights_path="lstmNishe.weights.h5", lookback=60, horizon=5):
    model = build_model(lookback, horizon)
    try:
        model.load_weights(weights_path)
        st.success("✅ Weights loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Could not load weights, using random init.\n\nError: {e}")
    return model

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 LSTM Stock Forecast — Close only")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2020,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = 60
horizon = 5

if st.sidebar.button("Run Prediction"):
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)

    last_seq = np.array([scaled[-lookback:]])  # shape (1, 60, 1)

    model = load_model("lstm.weights.h5", lookback, horizon)
    pred_scaled = model.predict(last_seq, verbose=0).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

    pred_index = pd.date_range(data.index[-1] + pd.Timedelta(days=1),
                               periods=horizon, freq="B")
    pred_df = pd.DataFrame({"Predicted Close": pred}, index=pred_index)

    st.write(pred_df)

    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label="Historical Close")
    plt.plot(pred_df.index, pred_df['Predicted Close'], marker="o", color="red", label="Forecast")
    plt.title(f"{ticker} — Close Forecast (5-day)")
    plt.legend()
    st.pyplot(plt.gcf())
