import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam




# to fetch the google stock data, downloading two years worth of data
google_data = yf.download('GOOG', period='2y')

# computing the moving averages of 5, 20, 50
google_data['MA_5'] = google_data['Close'].rolling(window=5).mean()
google_data['MA_20'] = google_data['Close'].rolling(window=20).mean()
google_data['MA_50'] = google_data['Close'].rolling(window=50).mean()

# our features
features = ['MA_5', "MA_20", "MA-50", "Clsoe"]

# Scaler is used to normalize data so the features are scalled between 0 and 1 for the LSTM performance
scaler = MinMaxScaler()
google_data_scaled = scalar.fit_transform(google_data[features]);

# prepping for output
# Creates a Target such that, if the price is higher than today, will output 1. 
google_data['Target'] = (google_data['Close'].shift(-1) > google_data['Close']).astype(int)
google_data.dropna(inplace=True)

# Spllitng data
x = google_data[:-1] # drop last row since there is no "tomorrow" to compare with
y = google_data['Target'].values[:-1]

# Spliting the data for training and testing sets
# test_size=0.2 for 20% testing and 80% training.
# random_state=42 to ensure reproducibility, consistency in split everytime.
# randomizes the splitting of taking the first 80%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True)

# configuing the data for the LSTM Model
# require a 3D input, (samples, time steps, features)
x_train = np.reshape(x_train, (x_train[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# LSTM Model
# sequential creates stacks of layers
model = Sequential([
    # 50 units, allows passing output to the next layer, 
    LSTM(50, return_sequences=True, input_shape=(1, x_train.shape[2])), 
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # 1 since binary classification, sigmoid is for probability from 0 to 1
])
# Loss Function, Optimizer, Metrics
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train Model
# train model for 20 full cycles. Updates weight after every 16 samples. Check's accuracy during training
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test))

# Test Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy *100:2f}%")