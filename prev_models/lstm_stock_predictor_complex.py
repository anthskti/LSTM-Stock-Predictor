import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input


# User input for stock ticker
ticker = input("Enter ticker: ").upper()

# checks if ticker exists
try:
    # to fetch the google stock data, downloading two years worth of data
    stock_data = yf.download(ticker, period='2y')
    if stock_data.empty:
        raise ValueError("Invalid ticker symbol or no data available.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# computing the moving averages of 5, 20, 50
stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()

# Golden Cross Feature, when the 5 day and 20 day SMA intersect
stock_data['Golden_Cross'] = (stock_data['MA_5'] > stock_data['MA_20']).astype(int)

# Relative Strength Index (RSI)
def calculate_rsi(stock_data, period=14):
    diff = stock_data['Close'].diff(1) # difference between the current day closing price and previous day
    gain = (diff.where(diff > 0, 0)).rolling(window=period).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=period).mean()
    rs = gain / loss # relative strength
    stock_data['RSI'] = 100 - (100/(1+rs)) # scaled between 0 and 100
    # RSI above 70 suggest overbought condition, potential price drop
    # RSI below 30 suggest an oversold condition, potential price increase
    return stock_data

stock_data = calculate_rsi(stock_data)

# Fibonacci Retracement Levels (using th past 60 days)
def calculate_fibonacci_levels(stock_data, period=60):
    stock_data['High_Max'] = stock_data['High'].rolling(window=period).max()
    stock_data['Low_Min'] = stock_data['Low'].rolling(window=period).min()
    stock_data['Fib_0.236'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.236
    stock_data['Fib_0.382'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.382
    stock_data['Fib_0.5'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.5
    stock_data['Fib_0.618'] = stock_data['High_Max'] - (stock_data['High_Max'] - stock_data['Low_Min']) * 0.618
    return stock_data

stock_data = calculate_fibonacci_levels(stock_data)
stock_data.dropna(inplace=True) # remove any rows

# our features 
features = ['MA_5', 'MA_20', 'MA_50', 'Golden_Cross', 'Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Volume', 'RSI', 'Close']

# Scaler is used to normalize data so the features are scalled between 0 and 1 for the LSTM performance
scaler = MinMaxScaler()
stock_data_scaled = scaler.fit_transform(stock_data[features]);

# prepping for output
# Instead of binary, we're predicting percentage change
stock_data['Target'] = ((stock_data['Close'].shift(-1) - stock_data['Close']) / stock_data['Close'])

# print("Before:")
# print(stock_data.tail(10))
stock_data.dropna(inplace=True)
# print("After")
# print(stock_data.tail(10))
# The most recent days stock_data['Target'] won't exist as there is no next day to compare it with

# Spllitng data
X = stock_data[features].iloc[:-1].values # Will explicitly select the four features
X = scaler.transform(X) 
y = stock_data['Target'].values[:-1]

# Spliting the data for training and testing sets
# test_size=0.2 for 20% testing and 80% training.
# random_state=42 to ensure reproducibility, consistency in split everytime.
# randomizes the splitting of taking the first 80%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

# configuing the data for the LSTM Model
# require a 3D input, (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# LSTM Model
# sequential creates stacks of layers
model = Sequential([
    Input(shape=(1, X_train.shape[2])),
    # 50 units, allows passing output to the next layer, 
    LSTM(100, return_sequences=True), 
    Dropout(0.3),
    LSTM(100),
    Dropout(0.3),
    Dense(50, activation='relu'), # since percentage
    Dense(1)
])
# Loss Function, Optimizer, Metrics (Mean Absolute Error)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

# Train Model
# train model for 20 full cycles. Updates weight after every 16 samples. Check's accuracy during training
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Test Model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae:.4f}%")

# For predictions of tomorrows stock 
# Preparing the todays data and reorganizing it to fit in the LSTM input
# Get the latest stock data for prediction
latest_data = stock_data[features].iloc[-1:].values # Extract last row
latest_data_scaled = scaler.transform(latest_data) # Scale the latest data using the same scaler
# Reshape data to match LSTM input shape: (samples=1, time steps=1, features=4)
latest_data_scaled = np.reshape(latest_data_scaled, (1, 1, latest_data_scaled.shape[1]))


# # Debug
# print("Shape of x_train:", X_train.shape)  # Should be (samples, 1, 4)
# print("Shape of latest_data_scaled:", latest_data_scaled.shape)  # Should be (1, 1, 4)
# print("Shape of df_scaled:", stock_data_scaled.shape)  # Should be (rows, 4), not (rows, 9)
# print(stock_data[features].head())  # Should show only 4 columns
# print("Scaled Latest Data: ", latest_data_scaled)
# print(stock_data.tail(10))

# tick = "GOOG"
# info = yf.download(tick, start="2025-02-10", end="2025-02-15")
# print(info)


# Make prediction
prediction = model.predict(latest_data_scaled)
# print(f"{prediction[0][0]}")
predicted_change = prediction[0][0] * 100
print(f'Predicted percent change for {ticker}: {predicted_change:.2f}%')



