# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import shap

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_lstm_model():
    return load_model("enhanced_lstm_stock.h5")

model = load_lstm_model()

# -----------------------
# Functions
# -----------------------
def add_indicators(df):
    """Add moving averages and exponential moving averages."""
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()
    return df.dropna()

def preprocess(df, lookback=90):
    """Scale data and create last lookback sequence."""
    features = ["Close", "Volume", "SMA_10", "SMA_30", "EMA_10", "EMA_30"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    last_seq = np.array([scaled[-lookback:]])  # shape (1, lookback, n_features)
    return last_seq, scaler, features

def forecast(model, seq, scaler, horizon=5):
    """Make prediction for next horizon days and inverse scale."""
    y_pred = model.predict(seq).reshape(horizon, 2)  # Close + Volume only
    # Pad back into feature space (other cols = 0) for inverse scaling
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
    seq, scaler, features = preprocess(df, lookback=lookback)
    pred_df = forecast(model, seq, scaler, horizon=horizon)

    # Assign dates to predictions (business days only)
    pred_df.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), 
                                  periods=horizon, freq="B")

    # -----------------------
    # Display Outputs
    # -----------------------
    st.subheader("Predicted Next 5 Days")
    st.write(pred_df)

    # Plot historical with SMA/EMA
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
        shap_fig = shap.plots.bar(shap_values[0], show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning("SHAP explanation not available in this environment.")
        st.text(str(e))

    # -----------------------
    # (Optional) ChatGPT Explanation
    # -----------------------
    st.subheader("AI-generated Explanation")
    st.info(
        "The model predicts the next 5 days of Close and Volume. "
        "Short-term EMAs vs long-term SMAs help highlight momentum trends. "
        "If EMA_10 > EMA_30, the model tends to predict upward momentum; "
        "if Volume is high while price is rising, it strengthens the bullish signal."
    )
