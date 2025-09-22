# app.py (Close + Volume version)

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
# Build SAME architecture as in Colab (Close + Volume)
# -----------------------------
def build_model(lookback=60, horizon=5):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 2)),  # 2 features
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(horizon * 2)  # 5 steps × (Close, Volume)
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
st.title("📊 LSTM Stock Forecast — Close & Volume")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2020,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = 60
horizon = 5

if st.sidebar.button("Run Prediction"):
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[['Close', 'Volume']].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)

    last_seq = np.array([scaled[-lookback:]])  # shape (1, 60, 2)

    model = load_model("lstmNishe.weights.h5", lookback, horizon)
    pred_scaled = model.predict(last_seq, verbose=0).reshape(horizon, 2)
    pad = np.zeros((horizon, scaler.n_features_in_))
    pad[:,:2] = pred_scaled
    inv = scaler.inverse_transform(pad)

    pred_df = pd.DataFrame(inv[:,:2], columns=["Predicted Close","Predicted Volume"],
                           index=pd.date_range(data.index[-1] + pd.Timedelta(days=1),
                                               periods=horizon, freq="B"))

    st.write(pred_df)

    fig, axes = plt.subplots(2,1,figsize=(12,10), sharex=True)
    axes[0].plot(data.index, data['Close'], label="Historical Close", color="blue")
    axes[0].plot(pred_df.index, pred_df['Predicted Close'], marker="o", color="red", label="Forecast Close")
    axes[0].legend(); axes[0].set_title("Close Forecast")

    axes[1].plot(data.index, data['Volume'], label="Historical Volume", color="blue")
    axes[1].plot(pred_df.index, pred_df['Predicted Volume'], marker="o", color="red", label="Forecast Volume")
    axes[1].legend(); axes[1].set_title("Volume Forecast")

    st.pyplot(fig)
