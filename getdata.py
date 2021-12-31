from numpy import *
import pandas as pd
import yfinance as yf
from numpy.random import rand, randn
from pandas.tseries.offsets import BDay

class Data:

    @classmethod
    def sigt(self, t):
        return 3/2+sin(2*(4*t-2))+2*exp(-16*(4*t-2)**2)

    @classmethod
    def sigx(self, x):
        return sqrt(abs(x))+0.5

    @classmethod
    def gen_data_t(self, n, equi=True):
        T = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)*sqrt(1/n)
        X = array([self.sigt(t)*db for t, db in zip(T, dB)])
        return T, X, dB

    @classmethod
    def gen_data_tx(self, n, equi=True):
        T = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)*sqrt(1/n)
        X = [0]
        for t, db in zip(T, dB):
            x = X[-1]
            X.append(x + self.sigt(t)*self.sigx(x)*db)
        return T, diff(X), dB

    @classmethod
    def gen_data_x(self, n, equi=True):
        T = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)*sqrt(1/n)
        X = [0]
        for db in dB:
            x = X[-1]
            X.append(x + self.sigx(x)*db)
        return T, diff(X), dB

    @classmethod
    def get_stock(self, n=None, start='2010-01-01', end='2021-03-01', ticker='AAPL'):
        a = yf.Ticker(ticker)
        if n:
            end = pd.Timestamp.today().floor('D')
            start = end-BDay(n)
            start = str(start).split()[0]
            end = str(end).split()[0]

        df = a.history(start=start, end=end).Close

        X = diff(array(df.apply(log)))
        n = len(X)
        T = linspace(0, 1, n)
        dB = array([0]*n)

        print(f'Using {ticker} data from {start} to {end} ({n} days)')
        
        return T, X, dB
