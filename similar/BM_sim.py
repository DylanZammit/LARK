#!/usr/bin/env python3

from numpy import *
from numpy.random import randn
import matplotlib.pyplot as plt

def sigx(x):
    return sqrt(abs(x))+0.5

def gen_data_x(n, a=0, b=1, equi=True):
    X = [0]
    for db in dB:
        x = X[-1]
        X.append(x + sigx(x)*db)
    return T, diff(X), dB

def gen_data_b(self, n, a=0, b=1, p=0.5):
    #mut = mu(T)

    B = cumsum(dB)
    phiB = sigx(B)

    dX = phiB*dB
    #dX = mut*dt + phiB*dB
    return T, dX, dB

n = 10000
a, b = 0, 10000
p = 1
dt = (b-a)/n
T = linspace(a, b, n)
dB = randn(n)*sqrt(dt)

#################################
T, dX, dB = gen_data_b(n, a, b, p)
X = cumsum(dX)
B = cumsum(dB)

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(T, dX, label='$dX=\mu_t dt + \phi(B_t)dB$')
ax[0].plot(T, dB, alpha=0.7, label='dB')
ax[1].plot(T, X, label='$dX=\mu_t dt + \phi(B_t)dB$')
ax[1].plot(T, B, alpha=0.7, label='B')

ax[2].plot(T, sigx(B), label='$\sigma(B_t)$')

ax[0].legend()
ax[1].legend()
#################################
T, dX, dB = gen_data_x(n, a, b, p)
X = cumsum(dX)
B = cumsum(dB)

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(T, dX, label='$dX=\mu_t dt + \phi(X_t)dB$')
ax[0].plot(T, dB, alpha=0.7, label='dB')
ax[1].plot(T, X, label='$dX=\mu_t dt + \phi(X_t)dB$')
ax[1].plot(T, B, alpha=0.7, label='B')

ax[2].plot(T, sigx(X), label='$\sigma(B_t)$')

ax[0].legend()
ax[1].legend()

plt.show()
