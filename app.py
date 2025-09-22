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
import tensorflow as tf

def explain_prediction(model, input_window):
    """
    Compute saliency scores for the last lookback window.
    input_window: shape (lookback, n_features)
    """
    input_tensor = tf.convert_to_tensor(
        input_window.reshape(1, input_window.shape[0], input_window.shape[1]),
        dtype=tf.float32
    )
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        # Focus only on the Close price of the first forecast day
        target = prediction[0, 0]  
    
    grads = tape.gradient(target, input_tensor).numpy()[0]
    grads_abs = np.abs(grads)
    grads_norm = grads_abs / (np.max(grads_abs) + 1e-8)  # normalize
    
    return grads_norm

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
# Streamlit App
# =====================================
st.title("📈 Stock Price & Volume Prediction with LSTM")
st.write("Lookback = 30 days, Horizon = 5 days (fixed)")

# Sidebar
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
forecast_days = 5 

# Fixed params
lookback = 30
horizon = 5
batch_size = 32
lr = 1e-3
optimizer = "adam"
epochs = 50

# Download data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
df = df[["Close", "Volume"]].dropna()

st.subheader(f"Data Preview ({ticker})")
st.dataframe(df.tail(5))

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df)
X, y = create_sequences(scaled, lookback, horizon)

train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model button
if st.button("Train Model 🚀"):
    set_seed(42)
    model = build_lstm_model(lookback, n_features=X.shape[2], horizon=horizon,
                             stack=(128,64), dropout=0.2, dense_units=64,
                             optimizer=optimizer, lr=lr)

    hist, y_pred_real, y_test_real, metrics = train_and_eval(
        model, X_train, y_train, X_test, y_test,
        batch_size=batch_size, max_epochs=epochs, scaler=scaler
    )

  

    # Predictions vs Actual
    pred_close = y_pred_real.reshape(-1,2)[:,0]
    true_close = y_test_real.reshape(-1,2)[:,0]
    pred_volume = y_pred_real.reshape(-1,2)[:,1]
    true_volume = y_test_real.reshape(-1,2)[:,1]

    test_index = df.index[train_size+lookback: train_size+lookback+len(pred_close)]

    # Fix mismatch
    min_len = min(len(test_index), len(true_close), len(pred_close))
    test_index = test_index[:min_len]
    true_close, pred_close = true_close[:min_len], pred_close[:min_len]
    true_volume, pred_volume = true_volume[:min_len], pred_volume[:min_len]

    fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=True)

    axes[0].plot(df.index[:train_size+lookback], df['Close'][:train_size+lookback], label="Train Close", color="green")
    axes[0].plot(test_index, true_close, label="Actual Close", color="blue")
    axes[0].plot(test_index, pred_close, label="Predicted Close", color="red")
    axes[0].set_title("Close Price Prediction")
    axes[0].legend()

    axes[1].plot(df.index[:train_size+lookback], df['Volume'][:train_size+lookback], label="Train Volume", color="green")
    axes[1].plot(test_index, true_volume, label="Actual Volume", color="blue")
    axes[1].plot(test_index, pred_volume, label="Predicted Volume", color="red")
    axes[1].set_title("Volume Prediction")
    axes[1].legend()

    st.pyplot(fig)

    # Forecast future
    st.subheader(f"🔮 Forecast for next 5 days")
    future_preds = forecast_future(model, scaled, lookback, horizon, scaler, steps=5)

    future_df = pd.DataFrame(future_preds, columns=["Close", "Volume"])
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5, freq="B")
    future_df.index = future_dates

    # Table with real dates
    st.dataframe(future_df)

    # Separate charts
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
        # Explainability
    st.subheader("🧠 Explainable AI — Feature Importance for Last 30 Days")
    last_window = scaled[-lookback:]  # last 30 days of scaled input
    importance = explain_prediction(model, last_window)

    importance_df = pd.DataFrame(
        importance, 
        columns=["Close_importance", "Volume_importance"], 
        index=df.index[-lookback:]
    )

    st.write("📊 Feature importance table (last 10 days shown):")
    st.dataframe(importance_df.tail(10))

    # Heatmap visualization
    fig4, ax = plt.subplots(figsize=(12,5))
    im = ax.imshow(importance.T, aspect="auto", cmap="Reds")
    ax.set_yticks([0,1])
    ax.set_yticklabels(["Close","Volume"])
    ax.set_xticks(range(lookback))
    ax.set_xticklabels(df.index[-lookback:].strftime("%Y-%m-%d"), rotation=90)
    ax.set_title("Saliency Heatmap — Importance of Past Days")
    fig4.colorbar(im, ax=ax)
    st.pyplot(fig4)







