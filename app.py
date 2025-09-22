# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# -----------------------
# Rebuild Final Model Architecture
# -----------------------
def build_final_model(lookback=90, n_features=2, horizon=5):
    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(lookback, n_features)),
        Dense(horizon * n_features)  # 5 × 2 = 10 outputs
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

@st.cache_resource
def load_trained_model(weights_path="lstm.weights.h5"):
    model = build_final_model()
    try:
        model.load_weights(weights_path)
        st.success("Model weights loaded successfully ✅")
    except Exception as e:
        st.warning("⚠️ Could not load weights. Running with random initialized weights.")
        st.text(str(e))
    return model

model = load_trained_model()

# -----------------------
# Helper Functions
# -----------------------
def preprocess(df, lookback=90):
    """Scale Close & Volume and return last lookback window."""
    features = ["Close", "Volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    last_seq = np.array([scaled[-lookback:]])  # (1, lookback, 2)
    return last_seq, scaler

def forecast(model, seq, scaler, horizon=5):
    """Forecast horizon steps for Close & Volume."""
    y_pred = model.predict(seq, verbose=0).reshape(horizon, 2)
    pad = np.zeros((horizon, scaler.n_features_in_))
    pad[:, :2] = y_pred
    inv = scaler.inverse_transform(pad)
    pred_df = pd.DataFrame(inv[:, :2], columns=["Close", "Volume"])
    return pred_df

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="Stock Forecast with LSTM", layout="wide")
st.title("📈 Stock Price & Volume Prediction (Enhanced LSTM)")

# Sidebar inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
lookback = st.sidebar.slider("Lookback (days)", 30, 180, 90)
horizon = 5

if st.sidebar.button("Run Prediction"):
    # -----------------------
    # Load Data
    # -----------------------
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df = df[["Close", "Volume"]].dropna()

    # -----------------------
    # Preprocess & Predict
    # -----------------------
    seq, scaler = preprocess(df, lookback=lookback)
    pred_df = forecast(model, seq, scaler, horizon=horizon)
    pred_df.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                                  periods=horizon, freq="B")

    # -----------------------
    # Display Outputs
    # -----------------------
    st.subheader("Predicted Next 5 Days")
    st.write(pred_df)

    # -----------------------
    # Plot Close & Volume
    # -----------------------
    fig, axes = plt.subplots(2, 1, figsize=(12,10), sharex=True)

    # Close
    axes[0].plot(df.index, df["Close"], label="Historical Close", color="green")
    axes[0].plot(pred_df.index, pred_df["Close"], marker="o", color="red", label="Predicted Close")
    axes[0].set_title(f"{ticker} — Close Price Forecast")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()

    # Volume
    axes[1].plot(df.index, df["Volume"], label="Historical Volume", color="blue")
    axes[1].plot(pred_df.index, pred_df["Volume"], marker="o", color="orange", label="Predicted Volume")
    axes[1].set_title(f"{ticker} — Volume Forecast")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Volume")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------
    # Narrative Explanation
    # -----------------------
    st.subheader("📊 Model Explanation")
    st.info(
        "This LSTM model was trained on 90 days of Close & Volume to predict the next 5 days. "
        "The top plot shows Close price forecasts, while the bottom plot shows Volume forecasts. "
        "When both Close and Volume rise together, it strengthens the bullish signal. "
        "When Close rises but Volume falls, the trend may be weaker."
    )
