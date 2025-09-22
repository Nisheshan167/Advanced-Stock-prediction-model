# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math, random, os, tensorflow as tf
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

    # Optimizer factory
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer.lower() == "adamw":
            try:
                opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
            except:
                opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=1e-4)
        elif optimizer.lower() == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss="mse")
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


# =====================================
# Streamlit App
# =====================================
st.title("📈 Stock Price & Volume Prediction with LSTM")
st.write("Enhanced LSTM model with lookback=30 days and horizon=5 days")

# Sidebar
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
lookback = 30
horizon = 5

# Model params
epochs = st.sidebar.slider("Epochs", 20, 200, 50, step=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lr = st.sidebar.selectbox("Learning Rate", [1e-3, 5e-4, 1e-4], index=0)
optimizer = st.sidebar.selectbox("Optimizer", ["adam", "adamw", "rmsprop", "sgd"], index=0)

# Download data
df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
df = df[["Close", "Volume"]].dropna()

st.subheader(f"Data Preview ({ticker})")
st.dataframe(df.head())

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

    st.success(f"✅ Model trained. RMSE Close: {metrics['RMSE_Close']:.2f}, RMSE Volume: {metrics['RMSE_Volume']:.2f}")

    # Plot predictions
    pred_close = y_pred_real.reshape(-1,2)[:,0]
    true_close = y_test_real.reshape(-1,2)[:,0]
    pred_volume = y_pred_real.reshape(-1,2)[:,1]
    true_volume = y_test_real.reshape(-1,2)[:,1]

    test_index = df.index[train_size+lookback: train_size+lookback+len(pred_close)]

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
