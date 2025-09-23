import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math, random, tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================================
# Utility functions
# =====================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:(i+lookback), :])
        y.append(data[(i+lookback):(i+lookback+horizon), :])
    return np.array(X), np.array(y)

def build_lstm_model(lookback, n_features, horizon,
                     stack=(128,64), dropout=0.2, dense_units=64,
                     optimizer="adam", lr=1e-3):
    model = Sequential()
    for li, units in enumerate(stack):
        return_sequences = (li < len(stack)-1)
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

    rmse_close = math.sqrt(mean_squared_error(y_test_real[:,:,0].ravel(), y_pred_real[:,:,0].ravel()))
    rmse_vol   = math.sqrt(mean_squared_error(y_test_real[:,:,1].ravel(), y_pred_real[:,:,1].ravel()))

    return hist, y_pred_real, y_test_real, {"RMSE_Close": rmse_close, "RMSE_Volume": rmse_vol}

def forecast_future(model, data, lookback, horizon, scaler, steps=5):
    last_window = data[-lookback:]
    preds = []
    current_input = last_window.copy()

    for _ in range(steps):
        X_input = np.expand_dims(current_input, axis=0)
        y_pred = model.predict(X_input, verbose=0).reshape(horizon, -1)
        preds.append(y_pred[0])   # take only first day of horizon
        current_input = np.vstack([current_input[1:], y_pred[0]])
    
    preds = np.array(preds)
    preds_real = scaler.inverse_transform(preds)
    return preds_real

# =====================================
# Integrated Gradients Explainability
# =====================================
def integrated_gradients(model, input_window, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(input_window)

    scaled_inputs = [baseline + (float(i)/steps)*(input_window-baseline) for i in range(steps+1)]
    grads = []

    for scaled in scaled_inputs:
        with tf.GradientTape() as tape:
            inp = tf.convert_to_tensor(scaled.reshape(1, *scaled.shape), dtype=tf.float32)
            tape.watch(inp)
            pred = model(inp)
            target = pred[0, 0]  # attribution wrt first forecasted Close price
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

# =====================================
# Streamlit App
# =====================================
st.title("📈 Short-Term Prediction of Stock Closing Prices and Market Volumes")

# Sidebar
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
df = df[["Close", "Volume"]].dropna()

# Indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

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
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df[["Close","Volume"]])
X, y = create_sequences(scaled, lookback, horizon)
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model button
if st.button("View Forecast 🚀"):
    set_seed(42)
    model = build_lstm_model(lookback, n_features=X.shape[2], horizon=horizon,
                             stack=(128,64), dropout=0.2, dense_units=64,
                             optimizer=optimizer, lr=lr)

    hist, y_pred_real, y_test_real, metrics = train_and_eval(
        model, X_train, y_train, X_test, y_test,
        batch_size=batch_size, max_epochs=epochs, scaler=scaler
    )

    st.subheader(f"🔮 Forecast for next 5 days")
    future_preds = forecast_future(model, scaled, lookback, horizon, scaler, steps=5)
    future_df = pd.DataFrame(future_preds, columns=["Close", "Volume"])
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq="B")
    future_df.index = future_dates
    st.dataframe(future_df)

    # Recommendation (force scalars)
    latest_close   = float(df['Close'].iloc[-1])
    forecast_price = float(future_df['Close'].iloc[0])
    sma20          = float(df['SMA_20'].iloc[-1])
    sma50          = float(df['SMA_50'].iloc[-1])
    rsi            = float(df['RSI'].iloc[-1])
    recommendation = stock_recommendation(latest_close, forecast_price, sma20, sma50, rsi)

    st.metric(label="Recommendation", value=recommendation)
    st.write(f"Latest Close: {latest_close:.2f}, Forecast Price (next day): {forecast_price:.2f}")

    # Charts
    fig2, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(future_df.index, future_df["Close"], marker="o", color="red")
    ax1.set_title("Forecasted Close Price")
    ax1.set_ylabel("Close Price")
    st.pyplot(fig2)

    fig3, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(future_df.index, future_df["Volume"], marker="o", color="blue")
    ax2.set_title("Forecasted Volume")
    ax2.set_ylabel("Volume")
    st.pyplot(fig3)

  
    # Integrated Gradients Explainability
    st.subheader("🧠 Explainable AI — Integrated Gradients (Last 30 Days)")
    last_window = scaled[-lookback:]  # last 30 days
    ig_attributions = integrated_gradients(model, last_window)
    ig_df = pd.DataFrame(
        ig_attributions,
        columns=["Close_importance", "Volume_importance"],
        index=df.index[-lookback:]
    )
    st.write("📊 Integrated Gradients attribution (last 10 days shown):")
    st.dataframe(ig_df.tail(10))
    fig4, ax = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ax[0].bar(ig_df.index, ig_df["Close_importance"], color="red")
    ax[0].set_title("Attribution for Close")
    ax[0].tick_params(axis="x", rotation=90)
    ax[1].bar(ig_df.index, ig_df["Volume_importance"], color="blue")
    ax[1].set_title("Attribution for Volume")
    ax[1].tick_params(axis="x", rotation=90)
    st.pyplot(fig4)

