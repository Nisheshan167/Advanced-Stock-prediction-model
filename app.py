import os
import math
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================
# OpenAI init + reporter
# ===============================
from openai import OpenAI

def _init_openai_client():
    key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:   # Streamlit Cloud secrets
            key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY")    # fallback for local dev

    if not key:
        st.warning("⚠️ OPENAI_API_KEY not found. GenAI analysis will be skipped.")
        return None

    return OpenAI(api_key=key)

# create global client once
client: OpenAI | None = _init_openai_client()

def generate_report(forecast_summary: str, indicators: str, recommendation: str) -> str:
    """LLM commentary on model outputs."""
    if client is None:
        return "ℹ️ GenAI commentary disabled (no API key found)."

    prompt = f"""You are a financial analyst.
Given the forecast summary: {forecast_summary},
indicators: {indicators},
and recommendation: {recommendation},
write a clear, professional 2–3 paragraph market commentary for an investor audience."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",   # or "gpt-4o", "gpt-4-turbo"
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating report: {e}"


# ===============================
# Utilities
# ===============================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i+lookback), :])
        y.append(data[(i+lookback):(i+lookback+horizon), :])
    return np.array(X), np.array(y)

def build_lstm_model(lookback, n_features, horizon,
                     stack=(128, 64), dropout=0.2, dense_units=64,
                     optimizer="adam", lr=1e-3):
    model = Sequential()
    for li, units in enumerate(stack):
        return_sequences = (li < len(stack) - 1)
        if li == 0:
            model.add(LSTM(units, return_sequences=return_sequences, input_shape=(lookback, n_features)))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        model.add(Dropout(dropout))
    if dense_units:
        model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(horizon * n_features))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def train_and_eval(model, X_train, y_train, X_test, y_test, batch_size=32, max_epochs=50, scaler=None):
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

    hist = model.fit(
        X_train, y_train.reshape(y_train.shape[0], -1),
        epochs=max_epochs, batch_size=batch_size, validation_split=0.1,
        callbacks=[es, rlrop], verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).reshape(-1, y_test.shape[1], y_test.shape[2])

    # Inverse scale
    y_pred_flat = y_pred.reshape(-1, 2)
    y_test_flat = y_test.reshape(-1, 2)
    y_pred_real = scaler.inverse_transform(y_pred_flat).reshape(y_pred.shape)
    y_test_real = scaler.inverse_transform(y_test_flat).reshape(y_test.shape)

    rmse_close = math.sqrt(mean_squared_error(y_test_real[:, :, 0].ravel(), y_pred_real[:, :, 0].ravel()))
    rmse_vol   = math.sqrt(mean_squared_error(y_test_real[:, :, 1].ravel(), y_pred_real[:, :, 1].ravel()))
    return hist, y_pred_real, y_test_real, {"RMSE_Close": rmse_close, "RMSE_Volume": rmse_vol}

def forecast_future(model, data, lookback, horizon, scaler, steps=5):
    last_window = data[-lookback:]
    preds = []
    current_input = last_window.copy()
    for _ in range(steps):
        X_input = np.expand_dims(current_input, axis=0)
        y_pred = model.predict(X_input, verbose=0).reshape(horizon, -1)
        preds.append(y_pred[0])   # first day of horizon
        current_input = np.vstack([current_input[1:], y_pred[0]])
    preds = np.array(preds)
    preds_real = scaler.inverse_transform(preds)
    return preds_real

def integrated_gradients(model, input_window, baseline=None, steps=50):
    """
    Integrated Gradients for LSTM inputs.
    input_window: shape (lookback, n_features)
    """
    if baseline is None:
        baseline = np.zeros_like(input_window)

    # Scale between baseline and input
    scaled_inputs = [baseline + (float(i)/steps)*(input_window-baseline) for i in range(steps+1)]
    grads = []

    for scaled in scaled_inputs:
        with tf.GradientTape() as tape:
            # Reshape to (1, lookback, n_features)
            inp = tf.convert_to_tensor(scaled.reshape(1, *scaled.shape), dtype=tf.float32)
            tape.watch(inp)
            pred = model(inp)
            # take first forecasted Close price as target
            target = pred[:, 0]  
        grads.append(tape.gradient(target, inp).numpy()[0])

    avg_grads = np.mean(grads, axis=0)
    integrated_grads = (input_window - baseline) * avg_grads
    return integrated_grads


def stock_recommendation(latest_close, forecast_price, sma20, sma50, rsi):
    if forecast_price > latest_close and sma20 > sma50 and rsi < 70:
        return "BUY"
    elif forecast_price < latest_close and rsi > 70:
        return "SELL"
    else:
        return "HOLD"


# ===============================
# App
# ===============================
st.title("📈 Short-Term Prediction of Stock Closing Prices and Market Volumes")

# Sidebar
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
df = df[["Close", "Volume"]].dropna()

# Indicators
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss.replace(0, np.nan) + 1e-9)
df["RSI"] = 100 - (100 / (1 + rs))

st.subheader(f"Data Preview ({ticker})")
st.dataframe(df.tail(5))

# Fixed params
lookback = 30
horizon = 5
batch_size = 32
lr = 1e-3
optimizer = "adam"
epochs = 50

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[["Close", "Volume"]])
X, y = create_sequences(scaled, lookback, horizon)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========= Single button (no duplicates) =========
if st.button("View Forecast 🚀"):
    set_seed(42)
    model = build_lstm_model(
        lookback, n_features=X.shape[2], horizon=horizon,
        stack=(128, 64), dropout=0.2, dense_units=64,
        optimizer=optimizer, lr=lr
    )

    hist, y_pred_real, y_test_real, metrics = train_and_eval(
        model, X_train, y_train, X_test, y_test,
        batch_size=batch_size, max_epochs=epochs, scaler=scaler
    )

    st.subheader("🔮 Forecast for next 5 days")
    future_preds = forecast_future(model, scaled, lookback, horizon, scaler, steps=5)
    future_df = pd.DataFrame(future_preds, columns=["Close", "Volume"])
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq="B")
    future_df.index = future_dates
    st.dataframe(future_df)

    # Recommendation inputs (scalars)
    latest_close   = float(df["Close"].iloc[-1])
    forecast_price = float(future_df["Close"].iloc[0])
    sma20          = float(df["SMA_20"].iloc[-1])
    sma50          = float(df["SMA_50"].iloc[-1])
    rsi            = float(df["RSI"].iloc[-1])
    recommendation = stock_recommendation(latest_close, forecast_price, sma20, sma50, rsi)

    st.metric(label="Recommendation", value=recommendation)
    st.write(f"Latest Close: {latest_close:.2f} | Forecast (next day): {forecast_price:.2f}")

    # Technical Indicators
    st.subheader("Technical Indicators")
    ti = pd.DataFrame(index=df.index)
    ti["Close"] = pd.to_numeric(df["Close"].squeeze(), errors="coerce")
    ti["SMA_20"] = ti["Close"].rolling(window=20, min_periods=20).mean()
    ti["SMA_50"] = ti["Close"].rolling(window=50, min_periods=50).mean()
    delta = ti["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss.replace(0, np.nan) + 1e-9)
    ti["RSI"] = 100 - (100 / (1 + rs))

    ma_plot = ti[["Close", "SMA_20", "SMA_50"]].dropna()
    rsi_plot = ti[["RSI"]].dropna()

    if ma_plot.empty or rsi_plot.empty:
        st.info("Not enough data to plot indicators yet (need ≥50 days for SMA_50 and ≥14 days for RSI).")
    else:
        st.line_chart(ma_plot)
        rsi_recent = rsi_plot.last("180D")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rsi_recent.index, rsi_recent["RSI"], label="RSI (14)")
        ax.axhline(70, linestyle="--", label="Overbought (70)")
        ax.axhline(30, linestyle="--", label="Oversold (30)")
        ax.set_title("Relative Strength Index (14) – Last 6 Months")
        ax.set_ylabel("RSI")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📖 Explanations")
    st.markdown("""
    - **SMA 20 vs SMA 50**: Short vs long-term momentum.
    - **RSI (14)**: >70 = Overbought, <30 = Oversold.
    - **Recommendation**: From model forecast + indicators.
    """)

    # Forecast charts
    fig2, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(future_df.index, future_df["Close"], marker="o")
    ax1.set_title("Forecasted Close Price")
    ax1.set_ylabel("Close Price")
    st.pyplot(fig2)

    fig3, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(future_df.index, future_df["Volume"], marker="o")
    ax2.set_title("Forecasted Volume")
    ax2.set_ylabel("Volume")
    st.pyplot(fig3)

    # Explainability (Integrated Gradients)
    st.subheader("🧠 Explainable AI — Integrated Gradients (Last 30 Days)")
    last_window = scaled[-lookback:]  # last 30 days
    ig_attributions = integrated_gradients(model, last_window)
    ig_df = pd.DataFrame(
        ig_attributions,
        columns=["Close_importance", "Volume_importance"],
        index=df.index[-lookback:]
    )
    st.write("📊 Integrated Gradients attribution (last 10 days):")
    st.dataframe(ig_df.tail(10))

    fig4, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].bar(ig_df.index, ig_df["Close_importance"])
    ax[0].set_title("Attribution for Close")
    ax[0].tick_params(axis="x", rotation=90)
    ax[1].bar(ig_df.index, ig_df["Volume_importance"])
    ax[1].set_title("Attribution for Volume")
    ax[1].tick_params(axis="x", rotation=90)
    st.pyplot(fig4)

    # ===============================
    # GenAI Natural-Language Analysis
    # ===============================
    st.subheader("📝 GenAI Analysis")
    five_day_avg = float(future_df["Close"].mean())
    indicators_text = (
        f"SMA20={sma20:.2f}, SMA50={sma50:.2f}, RSI={rsi:.1f}, "
        f"NextDayVolume={float(future_df['Volume'].iloc[0]):.0f}"
    )
    forecast_summary = (
        f"{ticker}: Next-day close {forecast_price:.2f} vs latest {latest_close:.2f}. "
        f"5-day mean path {five_day_avg:.2f}."
    )
    genai_report = generate_report(forecast_summary, indicators_text, recommendation)
    st.write(genai_report)

