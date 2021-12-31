import numpy as np
import pandas as pd
import yfinance as yf

def get_data(start='2010-01-01', end='2021-03-01', ticker='AAPL', use_daily=True):
    a = yf.Ticker(ticker)
    if use_daily:
        df = a.history(start=start, end=end).Close
    else:
        df = a.history(period='1mo', interval='5m')
        df = df.reset_index().Close

    X = df.apply(np.log)
    return np.array(X)

if __name__=='__main__':
    get_data()
