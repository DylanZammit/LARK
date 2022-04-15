from numpy import *
import pandas as pd
import yfinance as yf
from numpy.random import rand, randn
from pandas.tseries.offsets import BDay

def aexpon(x, y, p=1, s=0.05, side='right', **kwargs):
    if side=='right' and x < y: return 0
    if side=='left' and x > y: return 0
    if x < y: return 0
    return exp(-abs(x-y)**p/s)

class Data:

    @classmethod
    def jump(self, t):
        W = [-0.1, 0.2, 0.5, 0.9]
        P = [1.2, 1.6, 1.1, 1.2]
        s = 0.05
        B = [5, 2, 3, 1]
        return 0.1+sum([b*aexpon(x=t, y=w, p=p, s=s) for b, w, p in zip(B, W, P)])

    @classmethod
    def sigt(self, t):
        return 3/2+sin(2*(4*t-2))+2*exp(-16*(4*t-2)**2)

    @classmethod
    def sigx(self, x):
        return sqrt(abs(x)+0.5)

    @classmethod
    def gen_data_t(self, n, equi=True, mtype='sigt'):
        if n is None: n =1000
        sig = getattr(self, mtype, 'sigt')
        print('vol fun = {}'.format(sig.__name__))
        T = sorted(rand(n)) if not equi else linspace(0, 1, n)
        dB = randn(n)#*sqrt(1/n)
        X = array([sig(t)*db for t, db in zip(T, dB)])
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
    def gen_data_x(self, n, a=0, b=1, equi=True):
        T = sorted(rand(n)) if not equi else linspace(a, b, n)
        dt = (b-a)/n
        dB = randn(n)*sqrt(dt)
        X = [0]
        for db in dB:
            x = X[-1]
            X.append(x + self.sigx(x)*db)
        return T, diff(X), dB

    @classmethod
    def gen_data_b(self, n, a=0, b=1, p=0.5):
        T = linspace(a, b, n)
        dt = (b-a)/n
        #mut = mu(T)

        dB = randn(n)*sqrt(dt)
        B = cumsum(dB)
        phiB = self.sigx(B)

        dX = phiB*dB
        #dX = mut*dt + phiB*dB
        return T, dX, dB

    @classmethod
    def get_stock(self, n=None, start='2018-05-28', end='2022-03-26', ticker='AAPL', returns=True):
        a = yf.Ticker(ticker)
        if n:
            end = pd.Timestamp.today().floor('D')
            start = end-BDay(n)
            start = str(start).split()[0]
            end = str(end).split()[0]

        b = a.history(start=start, end=end)
        df = b.Close

        X = diff(array(df.apply(log))) if returns else diff(insert(array(df), 0, 0))
        n = len(X)
        T = linspace(0, 1, n)
        Treal = b.index[1:]

        X[X==0]=1e-8
        print(f'Using {ticker} data from {start} to {end} ({n} days)')
        
        return T, X, Treal

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    W = [-0.1, 0.2, 0.5, 0.9]
    P = [1.2, 1.6, 1.1, 1.2]
    s = 0.05
    B = [5, 2, 3, 1]

    dom = linspace(0, 1, 1000)
    plt.plot(dom, [Data.LARK_jump(x) for x in dom])

    for w, p, b in zip(W, P, B):
        plt.plot(dom, [aexpon(x=t, y=w, p=p, s=s) for t in dom], color='black', alpha=0.4)
    plt.show()

