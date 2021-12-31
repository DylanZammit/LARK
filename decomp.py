#!/usr/bin/env python3

import yfinance as yf
import argparse
import json
from time import time, sleep
import pdb
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import rand, randint, randn, exponential
from copy import deepcopy
from scipy.stats import poisson, gamma, norm

from common import *
from kernels import Kernels

class LARK(Kernels):

    def __init__(self, X, Y, p, kernel, **kwargs):
        '''
        kwargs are passed on as kernel parameters
        '''
        
        self.ap = 2
        self.bp = 0.75

        self.pb, self.pd, self.pu = p
        self.n = len(X)
        self.X, self.Y = X, Y
        self.cY = cumsum(Y)
        self.dt = 1/self.n # assuming equally spaced!!
        self.b0 = 1e-8

        K = getattr(self, kernel)
        self.K = K
        #self.K = lambda x, y: K(x, y, **kwargs)

        self.wa, self.wb = 1.2*min(self.cY), 1.2*max(self.cY)

    def init(self):
        a, b = 1, 1
        J = poisson.rvs(30) # what should this be??
        W = rand(J)
        B = gamma.rvs(a, scale=1/b, size=J)
        p = gamma.rvs(self.ap, scale=1/self.bp)

        J2 = poisson.rvs(30) # what should this be??
        W2 = rand(J2)*(self.wb-self.wa)+self.wa
        B2 = gamma.rvs(a, scale=1/b, size=J2)
        p2 = gamma.rvs(self.ap, scale=1/self.bp)
        return p, J, list(W), list(B), p2, J2, list(W2), list(B2)

    def nu(self, x, p, W, B):
        return self.b0 + sum([b*self.K(x, w, p=p) for w, b in zip(W, B)])

    def l(self, p, W, B, p2, W2, B2):
        l = 0
        for i in range(1, self.n):
            y0, y1 = self.cY[i-1], self.cY[i]
            t = self.X[i]
            nut = self.nu(t, p, W, B)
            nux = self.nu(y0, p2, W2, B2)
            l += norm.logpdf(y1, loc=y0, scale=sqrt(self.dt*nut*nux))
        return l

    def rj_mcmc_t(self, p, J, W, B, p2, J2, W2, B2):
        a, b = 1, 1
        p1 = p
        J1 = deepcopy(J)
        W1 = deepcopy(W)
        B1 = deepcopy(B)
        u = rand()

        l0 = self.l(p=p, W=W, B=B, p2=p2, W2=W2, B2=B2)

        if u < self.pb or J == 0: # birth
            J1 += 1
            w = rand()
            b = gamma.rvs(a, scale=1/b)
    
            W1.append(w)
            B1.append(b)

            l1 = self.l(p=p1, W=W1, B=B1, p2=p2, W2=W2, B2=B2)

            A1 = l1-l0
            A2 = log(self.pd)#-log(J)
            A3 = -log(self.pb)#+log(J1)
            A4 = A1+A2+A3
            A = min([0, A4])

        elif u < self.pb+self.pd: # death
            J1 -= 1
            j = randint(0, J)
            del W1[j], B1[j]
            
            l1 = self.l(p=p1, W=W1, B=B1, p2=p2, W2=W2, B2=B2)

            A1 = l1-l0
            A2 = log(self.pb)#-log(J)
            A3 = -log(self.pd)#+log(J1)
            A4 = A1+A2+A3
            A = min([0, A4])
            
        else: # update
            j = randint(0, J)
            w = rand()
            b = gamma.rvs(a, 1/b)
            W1[j] = w
            B1[j] = b

            l1 = self.l(p=p1, W=W1, B=B1, p2=p2, W2=W2, B2=B2)

            A1 = l1-l0
            A = min([0, A1])

        e = exponential(1)

        if e+A > 0:
            self.accepted+=1
            J, W, B = deepcopy(J1), deepcopy(W1), deepcopy(B1)

        return J, W, B

    def rj_mcmc_x(self, p, J, W, B, p2, J2, W2, B2):
        a, b = 1, 1
        p1 = p2
        J1 = deepcopy(J2)
        W1 = deepcopy(W2)
        B1 = deepcopy(B2)
        u = rand()

        l0 = self.l(p=p, W=W, B=B, p2=p2, W2=W2, B2=B2)

        if u < self.pb or J2 == 0: # birth
            J1 += 1
            #w = randn()
            w = rand()*(self.wb-self.wa)+self.wa
            b = gamma.rvs(a, scale=1/b)
    
            W1.append(w)
            B1.append(b)

            l1 = self.l(p=p, W=W, B=B, p2=p1, W2=W1, B2=B1)

            A1 = l1-l0
            A2 = log(self.pd)#-log(J)
            A3 = -log(self.pb)#+log(J1)
            A4 = A1+A2+A3
            A = min([0, A4])

        elif u < self.pb+self.pd: # death
            J1 -= 1
            j = randint(0, J2)
            del W1[j], B1[j]
            
            l1 = self.l(p=p, W=W, B=B, p2=p1, W2=W1, B2=B1)

            A1 = l1-l0
            A2 = log(self.pb)#-log(J)
            A3 = -log(self.pd)#+log(J1)
            A4 = A1+A2+A3
            A = min([0, A4])
            
        else: # update
            j = randint(0, J2)
            w = rand()*(self.wb-self.wa)+self.wa
            b = gamma.rvs(a, 1/b)
            W1[j] = w
            B1[j] = b

            l1 = self.l(p=p, W=W, B=B, p2=p1, W2=W1, B2=B1)

            A1 = l1-l0
            A = min([0, A1])

        e = exponential(1)

        if e+A > 0:
            self.accepted+=1
            J2, W2, B2 = deepcopy(J1), deepcopy(W1), deepcopy(B1)

        return J2, W2, B2

    def sample_lam(self, lam, p, W, B):
        lam1 = gamma.rvs(self.al, scale=1/self.bl)

        l0 = self.l(lam, p, W, B)
        l1 = self.l(lam1, p1, W, B)
        A1 = l1-l0+gamma.logpdf(lam, self.al, scale=1/self.bl)-gamma.logpdf(lam1, self.al, scale=1/self.bl)
        A = min([0, A1])

        e = exponential(1)

        if e+A > 0: lam = lam1
        return lam

    def sample_p(self, p, W, B):
        p1 = gamma.rvs(self.ap, scale=1/self.bp)

        l0 = self.l(p, W, B)
        l1 = self.l(p1, W, B)
        A1 = l1-l0+gamma.logpdf(p, self.ap, scale=1/self.bp)-gamma.logpdf(p1, self.ap, scale=1/self.bp)
        A = min([0, A1])

        e = exponential(1)

        if e+A > 0: p = p1
        return p

    def save(self, fn):
        out = {
            'X': list(self.X),
            'Y': list(self.Y),
            'post': list(self.res)
            }
        with open(fn, 'w') as f:
            json.dump(out, f)

    @timer
    def __call__(self, N=100, bip=0):
        self.accepted = 0
        res = []
        p, J, W, B, p2, J2, W2, B2 = self.init()
        p = 1.5
        p2 = 1.5

        for i in range(N):
            progress(i, N, 'LARK')
            J, W, B = self.rj_mcmc_t(p, J, W, B, p2, J2, W2, B2)
            J2, W2, B2 = self.rj_mcmc_x(p, J, W, B, p2, J2, W2, B2)
            #p = self.sample_p(p, W, B)
            if i > bip: res.append([p, J, W, B, p2, J2, W2, B2])

        self.res = res
        self.accept_pct = self.accepted/N*100
        print(f'\nAcceptence Ratio = {int(self.accept_pct)}%')
        return res

@timer
def plot_out(posterior, lark, pp=False, real=False):
    nu = lark.nu
    N = len(posterior)
    ps = []

    if 1:
        domt = linspace(0, 1, 1000)
        domx = linspace(min(lark.cY), max(lark.cY), 1000)

        if not real:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(domt, Data.sigt(domt)**2, label='$\sigma^2(t)$')
            ax[0].plot(lark.X, lark.Y, alpha=0.4, label='Observations', color='black')
            ax[1].plot(domx, Data.sigx(domx)**2, label='$\sigma^2(x)$')

        plot_post_t = []
        plot_post_x = []
        for i, post in enumerate(posterior):
            progress(i, N, 'Plotting')
            p, J, W, B, p2, J2, W2, B2 = post
            ps.append(p)

            plot_post_t.append([nu(t, p, W, B) for t in domt])
            plot_post_x.append([nu(y, p2, W2, B2) for y in domx])

            #if pp: plt.plot(dom, plot_post[-1], alpha=0.05, color='r')
        plot_post_t = matrix(plot_post_t)
        plot_post_x = matrix(plot_post_x)

        quantiles_t = []
        quantiles_x = []
        
        for i in range(1000):
            quantiles_t.append(quantile(plot_post_t[:,i], [0.025, 0.975]))
            quantiles_x.append(quantile(plot_post_x[:,i], [0.025, 0.975]))
        quantiles_t = matrix(quantiles_t)
        quantiles_x = matrix(quantiles_x)

        plot_post_mean_t = array(plot_post_t.mean(0))[0]
        plot_post_mean_x = array(plot_post_x.mean(0))[0]

        beta = (sum(sqrt(plot_post_mean_t))/1000)**2
        quantiles_t /= beta
        plot_post_mean_t /= beta
        quantiles_x *= beta
        plot_post_mean_x *= beta

        ax[0].plot(domt, plot_post_mean_t, label='Posterior Mean')
        ax[0].fill_between(domt, array(quantiles_t[:, 0].flatten())[0], array(quantiles_t[:, 1].flatten())[0], alpha=0.3, color='red')

        ax[1].plot(domx, plot_post_mean_x, label='Posterior Mean')
        ax[1].fill_between(domx, array(quantiles_x[:, 0].flatten())[0], array(quantiles_x[:, 1].flatten())[0], alpha=0.3, color='red')

        #plt.title(f'n={lark.n}, N={n}, kernel=$exp(-10|x-y|)$')
        plt.legend()
        plt.figure()

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(lark.X, lark.Y, alpha=0.4, label='Observations', color='black')
        ax[1].plot(lark.X, lark.cY, alpha=0.4, label='Observations', color='black')

    if 0:
        plt.figure()
        dom = linspace(0, max([3, max(ps)]), 1000)
        plt.plot(dom, [gamma.pdf(x, lark.ap, scale=1/lark.bp) for x in dom], label='prior')
        plt.hist(ps, bins=20, label='posterior', density=True)
        plt.legend()
    if 0:
        plt.figure()
        plt.plot([J for _, J, _, _ in posterior], label='J trace')
        plt.legend()
    if 0:
        # dXt = st * Zt; st is estimated
        # Zt = dXt/st
        # Pr(Zt<z) = Pr(dXt/st<z)
        dom = sorted(lark.cY)
        dB = lark.dB
        Bt = cumsum(dB)
        plt.figure()

        BM_post = []
        for i, post in enumerate(posterior):
            progress(i, N, 'Plotting')
            p, J, W, B = post
            ps.append(p)
            BM_post.append([nu(y, p, W, B) for y in dom])
        BM_post = matrix(BM_post)
        BM_post = lark.Y[None,:]/BM_post
        BM_post_mean = array(BM_post.mean(0))[0]
        quantiles = []
        for i in range(len(dom)): 
            quantiles.append(quantile(BM_post[:,i], [0.025, 0.975]))
        quantiles = matrix(quantiles)
        plt.plot(dom, Bt, label='True BM')
        plt.plot(dom, cumsum(BM_post_mean), label='Posterior Mean')
        lower = cumsum(array(quantiles[:,0].flatten())[0])
        upper = cumsum(array(quantiles[:,1].flatten())[0])
        plt.fill_between(dom, lower, upper, alpha=0.3, color='red')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    parser.add_argument('--n', help='Sample size', type=int, default=100)
    parser.add_argument('--N', help='MCMC iterations', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period', type=int, default=0)
    parser.add_argument('--noequi', help='equally spaced', action='store_true')
    parser.add_argument('--p', type=str, default='0.4,0.4,0.2')
    parser.add_argument('--real', help='Use real data', action='store_true')
    parser.add_argument('--ticker', type=str, help='ticker to get data', default='AAPL')
    parser.add_argument('--post', help='Plot posterior samples', action='store_true')
    parser.add_argument('--plot', help='Plot output', action='store_true')
    parser.add_argument('--plot_samples', help='Plot output', action='store_true')
    parser.add_argument('--save', type=str, help='file name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='Kernel function', choices=['haar', 'expon'], default='expon')
    args = parser.parse_args()

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    if args.real:
        X, Y, dB = Data.get_stock(n=args.n, ticker=args.ticker)
    else:
        X, Y, dB = Data.gen_data_t(args.n)
    lark = LARK(X=X, Y=Y, p=p, kernel=args.kernel)
    lark.dB = dB
    if not args.load:
        res = lark(N=args.N, bip=args.bip)
    else:
        with open(args.load) as fn:
            data = json.load(fn)
        res = data['post']
        lark.X = data['X']
        lark.Y = data['Y']
        lark.res = res

    if args.save: lark.save(args.save)
    if args.plot: plot_out(res, lark, args.plot_samples, args.real)

if __name__=='__main__':
    main()
