# stock_price_predictor/model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ta

def add_technical_indicators(df):
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd_diff()
    df.dropna(inplace=True)
    return df

def prepare_data(df, sequence_length=60, use_ohlcv=False, use_indicators=False):
    df = df.copy()

    if use_ohlcv:
        if use_indicators:
            df = add_technical_indicators(df)
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
        else:
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        features = ['Close']

    data = df[features].values
    target_index = features.index('Close')

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, target_index])

    X = np.array(X)
    y = np.array(y)

    if not use_ohlcv:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler, df

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_prices(model, X, scaler, use_ohlcv=False, use_indicators=False):
    predictions = model.predict(X)
    feature_count = X.shape[2]

    if feature_count > 1:
        padded = np.zeros((predictions.shape[0], feature_count))
        padded[:, feature_count - (3 if use_indicators else 2)] = predictions[:, 0]  # Close is at 3 or 2
        inv_transformed = scaler.inverse_transform(padded)
        return inv_transformed[:, padded.shape[1] - (3 if use_indicators else 2)]
    else:
        return scaler.inverse_transform(predictions).flatten()

def evaluate_model(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    return mse, rmse, mae

def forecast_future(model, last_sequence, days_ahead, scaler, use_ohlcv=False, use_indicators=False):
    feature_count = last_sequence.shape[1]
    current_sequence = last_sequence.copy()
    future_predictions = []

    for _ in range(days_ahead):
        pred = model.predict(current_sequence[np.newaxis, ...])[0, 0]

        if feature_count > 1:
            dummy = np.zeros((feature_count,))
            dummy[feature_count - (3 if use_indicators else 2)] = pred
            inv = scaler.inverse_transform([dummy])[0, feature_count - (3 if use_indicators else 2)]
        else:
            inv = scaler.inverse_transform([[pred]])[0, 0]

        future_predictions.append(inv)

        new_step = current_sequence[-1].copy()
        new_step[feature_count - (3 if use_indicators else 2)] = pred
        current_sequence = np.vstack([current_sequence[1:], new_step])

    return future_predictions
