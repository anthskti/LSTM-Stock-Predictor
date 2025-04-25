# LSTM-Stock-Predictor
Machine Learning Model for a Stock Share.\
Predicting the question: "Will tomorrow's price be higher than today's closing price?".

## Intuition 
My original idea was predicting the price of GOOG, but after achieving that, I kept adding more features.\
The stock market is full of potential. Investors and traders use pattern recognition to analyze and predict stock prices all the time. My intuition was to use those common stock patterns with machine learning to analyze a stock and conclude whether the price will go up or down.
Using:
- 5-day, 20-day, 50-day SMA (Simple Moving Averages) indicators
- the Golden Cross (as it can show large momentum)
- the RSI (Relative Strength Index)
- the Fibonacci Retracement Levels
- the VWAP (Volume Weighted Average Price)

I added and reorganized the data, then put it in an LSTM (long short-term memory) model for pattern recognition, predicting the next output, or in this case, the next day of the stock.\
Human predictions for stocks are usually 50/50, but this model can predict higher than that with all the indicators I've implemented within the script.\
There is an error rate calculator to show its potential success rate, as the stock market isn't just a yes or no, but it is a good baseline for those who are interested in investing.\
I also played around with showing the data, so using plotly, I convert the data into a simple line graph comparing real vs predicted.

### Prerequisites
Make Sure you have the following installed:
- [Python](https://www.python.org/)
- [Pip](https://pip.pypa.io/en/stable/) (Python package manager)

### Installation 
1. Clone Repository: 
``` bash
git clone https://github.com/anthskti/LSTM-Stock-Predictor.git
cd LSTM-Stock-Predictor
```

2. Enter in Python Virtual Environment and Install Packages (for macos)
``` bash 
python3 -m venv venv
source venv/bin/activate
pip install yfinance pandas numpy plotly scikit-learn tensorflow
```

3. Run Python Script for the following code
for the final version (including plotly)
``` bash
python3 ‎stock_predictor_modular.py
```
The other files are simply a means for comparison. However, the one listed above has all features.

Input any valid ticker and a prediction for tomorrows stock will show.
