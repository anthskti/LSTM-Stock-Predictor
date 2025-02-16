# LSTM-Google-Model
Machine Learning Model for GOOG Stock Share.\
Predicting the question: "Will tomorrow's price be higher than today's closing price?".

### Prerequisites
Make Sure you have the following install:
- [Python](https://www.python.org/)
- [Pip](https://pip.pypa.io/en/stable/) (Python package manager)

### Installation 
1. Clone Repository: 
``` bash
git clone https://github.com/anthskti/LSTM-Google-Model.git
cd LSTM-Google-Model
```

2. Enter in Python Virtual Environment and Install Packages (for macos)
``` bash 
python3 -m venv venv
source venv/bin/activate
pip install yfinance pandas numpy plotly scikit-learn tensorflow
```

3. Run Python Script 
``` bash
python3 lstm_google_stock.py
```

Input any valid ticker and a prediction for tomorrows stock will show.
