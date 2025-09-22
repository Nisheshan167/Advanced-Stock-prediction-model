# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import shap
import os

# -----------------------
# Rebuild Final Model Architecture
# -----------------------
def build_final_model(lookback=90, n_features=6, horizon=5):
    model = Sequential([
        LSTM(64, input_shape=(lookback, n_features)),  # LSTM 64 units
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(horizon * 2)  # 10 units (5 days × 2 outputs)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

@st.cache_resource
def load_trained_model(weights_path="lstm.weights.h5"):
    model = build_final_model()
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            st.success("✅ Trained weights loaded successfully!")
        except Exception as e:
            st.warning("⚠️ Could not load weights, using untrained model instead.")
            st.text(str(e))
    else:
        st.warning(f"⚠️ Weights file '{weights_path}' not found. Using untrained model.")
    return model

model = load_trained_model()

# -----------------------
# Helper Functions
# -----------------------
def add_indicators(df):
    """Add SMA and EMA indicators to dataframe."""
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
    return df.dropna()

def preprocess(df, lookback=90):
    """Scale data and create last lookback sequence with 6 features."""
    features = ["Close", "Volume", "SMA_10", "SMA_30", "EMA_10", "EMA_30"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    last_seq = np.array([scaled[-lookback:]])  # shape (1, lookback, 6)
    return last_seq, scaler

def forecast(model, seq, scaler, horizon=5):
    """Make forecast and inverse transform."""
    y_pred = model.predict(seq).reshape(horizon, 2)  # Close + Volume
    pad = np.zeros((y_pred.shape[0], scaler.n_features_in_))
    pad[:, 0] = y_pred[:, 0]  # Close
    pad[:, 1] = y_pred[:, 1]  # Volume
    inv = scaler.inverse_transform(pad)
    pred_df = pd.DataFrame(inv[:, :2], columns=["Close", "Volume"])
    return pred_df

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="Stock Forecast with LSTM", layout="wide")
st.title("📈 Stock Price Prediction (LSTM + Explainable AI)")

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
    df = add_indicators(df)

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

    # Plot with indicators
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["Close"], label="Historical Close", color="green")
    ax.plot(df.index, df["SMA_10"], label="SMA 10", linestyle="--", color="orange")
    ax.plot(df.index, df["SMA_30"], label="SMA 30", linestyle="--", color="blue")
    ax.plot(df.index, df["EMA_10"], label="EMA 10", linestyle=":", color="red")
    ax.plot(df.index, df["EMA_30"], label="EMA 30", linestyle=":", color="purple")
    ax.plot(pred_df.index, pred_df["Close"], marker="o", color="black", label="Predicted Close")
    ax.set_title(f"{ticker} Close Price with Indicators & Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # -----------------------
    # Explainable AI (SHAP)
    # -----------------------
    st.subheader("Explainable AI (SHAP Feature Importance)")
    try:
        explainer = shap.Explainer(model, seq)
        shap_values = explainer(seq)
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning("⚠️ SHAP explanation not available in this environment.")
        st.text(str(e))

    # -----------------------
    # Static Explanation
    # -----------------------
    st.subheader("AI-generated Narrative")
    st.info(
        "The model predicts the next 5 days of Close and Volume. "
        "When short-term EMAs rise above long-term EMAs (e.g., EMA_10 > EMA_30), "
        "it indicates bullish momentum. If Volume also increases alongside rising prices, "
        "the upward trend is considered stronger."
    )
