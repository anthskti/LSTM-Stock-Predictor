# Trace
Machine Learning Model for a Stock Share.\
Predicting the question: "Will tomorrow's price be higher than today's closing price?".

My original Idea started with, predicting the price of GOOG, but ended up making it more versatile.\
The stock market is full of potential, using pattern recognition is something investors and traders use to analyze and predict stock prices. My intuition was to use those common stock patterns with machine learning to analyze a stock and conclude if the price with go up or down.
Using 5 day, 20 day, and 50 day SMA (Simple Moving Averages) indicators with the Golden Cross (as it can show large momentum) and the Fibonnaci Retracement Levels, I reorganized and inputted the data of the desired stock into an LSTM (long short-term memory) model to find patterns and predict the next day stock.\
There is a 50% chance you can be right if you look at a stock, but this model can predict higher than that with all the indicators I've implemented within the script.\
Though it is not always right, as the stock market is always flucuating, it is a good baseline for those who are interested in investing.

### Prerequisites
Make Sure you have the following install:
- [Python](https://www.python.org/)
- [Pip](https://pip.pypa.io/en/stable/) (Python package manager)

### Installation 
1. Clone Repository: 
``` bash
git clone https://github.com/anthskti/Trace.git
cd Trace
```

2. Enter in Python Virtual Environment and Install Packages (for macos)
``` bash 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run Python Script for the following code
``` bash
python3 trace-main.py
```
I kept the other models I had for comparison, feel free to compare looking at the "prev_models" folder. 

Input any valid ticker and a prediction for tomorrows stock will show.
