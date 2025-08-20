# stock_price_predictor/app.py

import streamlit as st
import yfinance as yf
from model import (
    prepare_data, build_lstm_model, predict_prices,
    evaluate_model, forecast_future
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor using LSTM")

st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
use_ohlcv = st.sidebar.radio("Feature Set", ["Close Only", "OHLCV"]) == "OHLCV"
use_indicators = False
if use_ohlcv:
    use_indicators = st.sidebar.checkbox("Include Technical Indicators (RSI, MACD)", value=True)

forecast_days = st.sidebar.slider("Days to Forecast into the Future", min_value=1, max_value=30, value=7)

if st.sidebar.button("Predict"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found. Please check the ticker symbol or date range.")
        else:
            st.subheader(f"Closing Price of {ticker}")
            st.line_chart(data['Close'])

            X, y, scaler, clean_df = prepare_data(data, use_ohlcv=use_ohlcv, use_indicators=use_indicators)
            model = build_lstm_model((X.shape[1], X.shape[2]))

            with st.spinner("Training the model..."):
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)

            predictions = predict_prices(model, X, scaler, use_ohlcv=use_ohlcv, use_indicators=use_indicators)
            actual = clean_df['Close'].values[-len(predictions):]

            # Evaluation metrics
            mse, rmse, mae = evaluate_model(actual, predictions)
            st.subheader("📊 Model Evaluation Metrics")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")
            st.write(f"**MAE:** {mae:.4f}")

            # Plot predictions vs actual
            st.subheader("Actual vs Predicted Prices")
            fig, ax = plt.subplots()
            ax.plot(actual, label='Actual')
            ax.plot(predictions, label='Predicted')
            ax.legend()
            st.pyplot(fig)

            # Forecast future prices
            st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
            last_seq = X[-1]  # Use last known sequence
            future_preds = forecast_future(
                model, last_seq, forecast_days,
                scaler, use_ohlcv=use_ohlcv, use_indicators=use_indicators
            )

            future_dates = pd.date_range(start=clean_df.index[-1], periods=forecast_days + 1, freq='B')[1:]
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Close': future_preds})
            st.line_chart(forecast_df.set_index('Date'))

    except Exception as e:
        st.error(f"An error occurred: {e}")
