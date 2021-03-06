#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincc
from scipy.stats import invgamma
from scipy.optimize import fsolve
import pandas as pd
import yfinance as yf
from getdata import Data
from common import RMSE

def gaus_ker(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

class Gugu:
    def __init__(self, Y, alpha=0.1, beta=0.1, m=30):
        self.Y = pd.Series(Y)
        n = len(Y)
        self.N = n//m
        self.m = m
        self.r=n-m*self.N
        self.n = n
        self.h=m/(2*n)

        self.alpha = alpha
        self.beta = beta

    def s2_kernel(self, t): #boxcar function
        n = self.n
        h = self.h
        m = self.m

        mid = int(t*n)
        l, r = m//2, m//2 #check if this correct
        inds = list(range(max([0, mid-l]), min([mid+r, n-1])))
        return 1/(2*h)*(self.Y.iloc[inds]**2).sum()

    def s2_kernel_gauss(self, t): #gauss kernel
        n = self.n
        h = self.h
        m = self.m
        return sum(gaus_ker((t-i/n)/h)*y**2 for i, y in enumerate(self.Y.fillna(0)))/h

    def s2_gugu(self):
        N = self.N
        m = self.m
        r = self.r
        Y = self.Y
        alpha = self.alpha
        beta = self.beta
        n = self.n

        z = [(Y.iloc[m*k:m*(k+1)]**2).sum() for k in range(N-1)]
        z.append((Y.iloc[m*(N-1):]**2).sum())

        alpha_1 = [alpha+m/2]*len(z[:-1]) + [alpha+(m+r)/2]
        beta_1 = [beta+n*zk/2 for zk in z]

        mean = [b/(a-1) for a,b in zip(alpha_1, beta_1)]
        upper_95 = [invgamma.ppf(0.975, a, scale=b) for a,b in zip(alpha_1, beta_1)]
        lower_95 = [invgamma.ppf(0.025, a, scale=b) for a,b in zip(alpha_1, beta_1)]    

        reps = [m]*(N-1)+[r+m]
        mean = np.repeat(mean, reps)
        upper_95 = np.repeat(upper_95, reps)
        lower_95 = np.repeat(lower_95, reps)

        return np.sqrt(mean), np.sqrt(lower_95), np.sqrt(upper_95)

def plot_gugu(model, T, gentype='sigt', Treal=None):
    X = model.Y
    dom = np.linspace(0, 1, len(X))
    n = len(X)

    mean, low, up= model.s2_gugu()
    mean, low, up = mean/np.sqrt(n), low/np.sqrt(n), up/np.sqrt(n)
    if gentype !='real': 
        truevol = [getattr(Data, gentype)(x) for x in dom]
        plt.plot(dom, truevol, label='True volatility', color='orange')
    if Treal is not None:
        Tdom = Treal
        plt.xticks(rotation=45)
    else:
        Tdom = T
    plt.plot(Tdom, mean, label='Histogram-Type', color='C0')
    plt.fill_between(Tdom, low, up, alpha=0.5, color='C0')
    #plt.plot(df.index, [model.s2_kernel(t) for t in dom], label='kernel_boxcar')
    #plt.plot(T, [model.s2_kernel_gauss(t) for t in dom], label='kernel_gauss')
    plt.plot(Tdom, model.Y, color='black', alpha=0.5, label='Obeservations')
    plt.legend()
    #RMSE################
    if gentype!='real':
        A = np.array([getattr(Data, gentype)(x) for x in T])
        B = mean
        print('\nGUGU RMSE = {}'.format(RMSE(A, B)))
    #RMSE################

def main():
    a = yf.Ticker("AAPL")

    use_daily = True

    if use_daily:
        df = a.history(start='2010-01-01', end='2021-03-01').Close
    else:
        df = a.history(period='1mo', interval='5m')

    df = df.reset_index().Close
    X = df.apply(np.log).diff()
    model = Gugu(X)
    plot_gugu(model, df.index)
    plt.figure()
    df.plot()
    plt.show()

if __name__ == '__main__':
    main()
