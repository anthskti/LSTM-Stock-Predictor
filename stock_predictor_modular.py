import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import logging
import os

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetching stock data from Yahoo Finance.
def fetch_stock_data(ticker, period='2y'):
    try:
    # to fetch the google stock data, downloading two years worth of data
        stock_data = yf.download(ticker, period=period)
        if stock_data.empty:
            raise ValueError("Invalid ticker symbol or no data available.")
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching stockdata: {e}")
        exit()

def calculate_moving_averages(stock_data):
    stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    return stock_data

def calculate_golden_cross(stock_data):
    stock_data['Golden_Cross'] = (stock_data['MA_5'] > stock_data['MA_20']).astype(int)
    return stock_data

# 
def calculate_rsi(stock_data, period=14):
    diff = stock_data['Close'].diff(1) # difference between the current day closing price and previous day
    gain = (diff.where(diff > 0, 0)).rolling(window=period).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=period).mean()
    rs = gain / loss # relative strength
    stock_data['RSI'] = 100 - (100/(1+rs)) # scaled between 0 and 100
    # RSI above 70 suggest overbought condition, potential price drop
    # RSI below 30 suggest an oversold condition, potential price increase
    return stock_data

# Fibonacci Retracement Levels (using th past 60 days)
def calculate_fibonacci_levels(stock_data, period=60):
    stock_data['High_Max'] = stock_data['High'].rolling(window=period).max()
    stock_data['Low_Min'] = stock_data['Low'].rolling(window=period).min()
    stock_data['Fib_0.236'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.236
    stock_data['Fib_0.382'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.382
    stock_data['Fib_0.5'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.5
    stock_data['Fib_0.618'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.618
    return stock_data

# VWAP / Volume Weighted Average Price
def calculate_vwap(stock_data):
    weighted_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
    stock_data['VWAP'] = (weighted_price * stock_data['Volume']).cumsum() + stock_data['Volume'].cumsum()
    return stock_data

# Prepping data for the training model
def prepare_data(stock_data, features):
    # Scaler is used to normalize data so the features are scalled between 0 and 1 for the LSTM performance
    scaler = MinMaxScaler()
    stock_data_scaled = scaler.fit_transform(stock_data[features]);
    # prepping for output
    # Instead of binary, we're predicting percentage change
    stock_data['Target'] = ((stock_data['Close'].shift(-1) - stock_data['Close']) / stock_data['Close'])
    stock_data.dropna(inplace=True)
    # Spllitng data
    X = stock_data[features].iloc[:-1].values # Will explicitly select the four features
    X = scaler.transform(X) 
    y = stock_data['Target'].values[:-1]
    return X, y, scaler

# Build and compile the LSTM model.
def build_lstm_model(input_shape):
    # LSTM Model
    # sequential creates stacks of layers
    model = Sequential([
        Input(shape=input_shape),
        # 100 units, allows passing output to the next layer, 
        LSTM(100, return_sequences=True), 
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation='relu'), # since percentage
        Dense(1)
    ])
    # Loss Function, Optimizer, Metrics (Mean Absolute Error)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    return model

# Actual vs Predicted on Plotly
def plot_results(stock_data, y_test, y_pred):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_test, mode='lines', name='Actual', line=dict(color='blue')))
    figure.add_trace(go.Scatter(x=stock_data.index[-len(y_test):], y=y_pred.flatten(), mode='lines', name='Predicted', line=dict(color='red')))
    figure.update_layout(title="Actual vs Predicted Percentage Change", xaxis_title="Date", yaxis_title="Percentage Change")
    figure.show()

# Load and Save Model Feature to save usage of yFinance requests
def load_model_from_file(filename="stock_predictor_model.h5"):
    if os.path.exists(filename):
        model = load_model(filename)
        logging.info(f"Model loaded from {filename}")
        return model
    else:
        logging.error(f"Model file {filename} not found.")
        exit()

def save_model_to_file(model, filename="stock_predictor_model.h5"):
    save_model(model, filename)
    logging.info(f"Model saved to {filename}")


def main():
    # User input for stock ticker
    ticker = input("Enter ticker: ").upper()
    
    # Fetching stock data
    stock_data = fetch_stock_data(ticker)

    # Calcuating features
    stock_data = calculate_moving_averages(stock_data)
    stock_data = calculate_golden_cross(stock_data)
    stock_data = calculate_rsi(stock_data)
    stock_data = calculate_fibonacci_levels(stock_data)
    stock_data = calculate_vwap(stock_data)
    stock_data.dropna(inplace=True)

    # Defining our features
    features = ['MA_5', 'MA_20', 'MA_50', 'Golden_Cross', 'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Volume', 'RSI', 'VWAP', 'Close']

    # Preping Data
    X, y, scaler = prepare_data(stock_data, features)

    # Spliting the data for training and testing sets
    # test_size=0.2 for 20% testing and 80% training.
    # random_state=42 to ensure reproducibility, consistency in split everytime.
    # randomizes the splitting of taking the first 80%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Reshaping the data for the LSTM Model
    # require a 3D input, (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


    model_filename = 'stock_predictor_model.h5'
    if os.path.exists(model_filename):
        model = load_model_from_file()
    else:
        model = build_lstm_model((1, X_train.shape[2]))
        # train model for 20 full cycles. Updates weight after every 16 samples. Check's accuracy during training
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))
        save_model_to_file(model, model_filename)

    # Test Model
    loss, mae = model.evaluate(X_test, y_test)
    logging.info(f"Test Mean Absolute Error: {mae:.4f}%")

    # For predictions of tomorrows stock 
    y_pred = model.predict(X_test)
    plot_results(stock_data, y_test, y_pred)

    # Preparing the todays data and reorganizing it to fit in the LSTM input
    # Get the latest stock data for prediction
    latest_data = stock_data[features].iloc[-1:].values # Extract last row
    latest_data_scaled = scaler.transform(latest_data) # Scale the latest data using the same scaler
    # Reshape data to match LSTM input shape: (samples=1, time steps=1, features=4)
    latest_data_scaled = np.reshape(latest_data_scaled, (1, 1, latest_data_scaled.shape[1]))
    # Make prediction
    prediction = model.predict(latest_data_scaled)
    # print(f"{prediction[0][0]}")
    predicted_change = prediction[0][0] * 100
    print(f'Predicted percent change for {ticker}: {predicted_change:.2f}%')

if __name__ == "__main__":
    main()


