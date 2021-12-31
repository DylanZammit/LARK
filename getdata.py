from numpy import *
import pandas as pd
import yfinance as yf
from numpy.random import rand, randn

class Data:

    @classmethod
    def sigt(self, t):
        return 3/2+sin(2*(4*t-2))+2*exp(-16*(4*t-2)**2)

    @classmethod
    def sigx(self, x):
        return sqrt(abs(x))+0.5

    @classmethod
    def gen_data_t(self, n, equi=True):
        X = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)
        Y = array([self.sigt(x)*e for x, e in zip(X, dB)])
        return X, Y, dB

    @classmethod
    def gen_data_tx(self, n, equi=True):
        X = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)*sqrt(1/n)
        Y = [0]
        for i in range(n):
            db = dB[i]
            y0 = Y[-1]
            x0 = X[i]
            Y.append(y0 + self.sigt(x0)*self.sigx(y0)*db)
        return X, diff(Y), dB

    @classmethod
    def gen_data_x(self, n, equi=True):
        X = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)*sqrt(1/n)
        Y = [0]
        for db in dB:
            y0 = Y[-1]
            Y.append(y0 + self.f(y0)*db)
        return X, diff(Y), dB

    @classmethod
    def get_stock(self, n=None, start='2010-01-01', end='2021-03-01', ticker='AAPL'):
        print(f'Using {ticker} data from {start} to {end}')
        a = yf.Ticker(ticker)
        if n:
            end = pd.Timestamp.today().floor('D')
            start = end-pd.Timedelta(f'{n}D')
            start = str(start).split()[0]
            end = str(end).split()[0]

        df = a.history(start=start, end=end).Close

        X = diff(array(df.apply(log)))
        T = linspace(0, 1, len(X))
        dB = array([0]*len(X))
        return T, X, dB
