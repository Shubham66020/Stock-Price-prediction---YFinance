# 📈 Stock Price Predictor using LSTM

This project is a stock price prediction app powered by deep learning. It uses an LSTM (Long Short-Term Memory) neural network to analyze historical stock data and forecast future prices. The app is interactive and built with Streamlit, allowing users to visualize and forecast prices with customizable inputs and feature selections.

---

## Features

-  Predicts prices using LSTM neural networks
-  Uses OHLCV (Open, High, Low, Close, Volume) data
-  Optionally includes technical indicators: RSI & MACD
-  Forecasts **future N-day** stock prices using a sliding window
-  Visualizes actual vs predicted prices and evaluation metrics
-  Simple Streamlit interface with intuitive controls
-  Evaluation metrics: MSE, RMSE, MAE

---

## 🖥️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Machine Learning**: TensorFlow/Keras (LSTM)
- **Data**: Yahoo Finance via `yfinance`
- **Technical Indicators**: `ta` (Technical Analysis library)
- **Visualization**: Matplotlib, Pandas

---

## Installation

1. **Clone the repository**:
  
   git clone https://github.com/your-username/stock-price-predictor.git
   cd stock-price-predictor

2. **Install Dependencied**
   pip install -r requirements.txt

3. **Run the app**
   streamlit rum app.py

4. **Requirements**
    streamlit
    yfinance
    pandas
    numpy
    matplotlib
    tensorflow
    scikit-learn
    ta

5. **Screenshots**

    **HomeScreen**
    ![![Homepage](./ScreenShots/homepage.png)]

    **Actual VS Predicted Chart**
    ![![Chart](./ScreenShots/actual_vs_predicted_chart.png)]

    **Forecasting**
    ![![Forecasting](./ScreenShots/forecasting.png)]

6.  **How it Works**

    **User Inputs:**

    Ticker symbol (e.g. AAPL, RELIANCE.NS)

    Date range

    Feature set: Close-only or OHLCV

    Toggle technical indicators (RSI, MACD)

    Days to forecast (1–30)

    **Data Processing:**

    Download historical data using yfinance

    Normalize using MinMaxScaler

    Optional indicators added using ta

    **Model Training:**

    LSTM layers + Dropout for regularization

    Trained on past 60-day windows

    Prediction & Forecasting:

    Predict current trend from test set

    Forecast unseen future prices using iterative window-based logic

7. **Folder Structure**
    stock-price-predictor/
    │
    ├── app.py               # Streamlit UI
    ├── model.py             # All ML/data functions
    ├── requirements.txt     # All Python dependencies
    ├── README.md            
    └── assets/              # screenshots

MIT License © 2025 Shubham Kumar
