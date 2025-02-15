import yfinance as yf

# to fetch the google stock data
google_stock_data = yf.download('GOOG', period='2y')