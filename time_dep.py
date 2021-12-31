#!/usr/bin/env python3

import argparse
import json
from time import time
import pdb
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import rand, randint, randn, exponential
from copy import deepcopy
from scipy.stats import poisson, gamma

from common import *
from kernels import Kernels

class Data:

    @classmethod
    def f(self, x):
        return (3 + sin(2*pi*x))/2
        #return (1.02 + sin(2*pi*x))/2+1.5*(x>=0.5)

    @classmethod
    def gen_data(self, n):
        X = sorted(rand(n))
        err = randn(n)
        Y = array([self.f(x)*e for x, e in zip(X, err)])
        return X, Y

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
        self.b0 = 1e-8

        K = getattr(self, kernel)
        self.K = K
        #self.K = lambda x, y: K(x, y, **kwargs)

    def init(self):
        a, b = 1, 1
        J = poisson.rvs(10) # what should this be??
        W = rand(J)
        B = gamma.rvs(a, scale=1/b, size=J)
        p = gamma.rvs(self.ap, scale=1/self.bp)
        return p, J, list(W), list(B)

    def nu(self, x, p, W, B):
        return self.b0 + sum([b*self.K(x, w, p=p) for w, b in zip(W, B)])

    def l(self, p, W, B):

        nus = [self.nu(x, p, W, B) for x in self.X]

        T1 = -self.n*log(2*pi)
        T2 = -sum([log(nui) for nui in nus])
        T3 = -sum([y**2/nui for nui, y in zip(nus, self.Y)])
        return (T1+T2+T3)/2

    def rj_mcmc(self, p, J, W, B):
        a, b = 1, 1
        p1 = p
        J1 = deepcopy(J)
        W1 = deepcopy(W)
        B1 = deepcopy(B)
        u = rand()

        l0 = self.l(p, W, B)

        if u < self.pb or J == 0: # birth
            J1 += 1
            w = rand()
            b = gamma.rvs(a, scale=1/b)
    
            W1.append(w)
            B1.append(b)

            l1 = self.l(p1, W1, B1)

            A1 = l1-l0
            A2 = log(self.pd)#-log(J)
            A3 = -log(self.pb)#+log(J1)
            A4 = A1+A2+A3
            A = min([0, A4])

        elif u < self.pb+self.pd: # death
            J1 -= 1
            j = randint(0, J)
            del W1[j], B1[j]
            
            l1 = self.l(p, W1, B1)

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

            l1 = self.l(p, W1, B1)

            A1 = l1-l0
            A = min([0, A1])

        e = exponential(1)

        if e+A > 0:
            self.accepted+=1
            J, W, B = deepcopy(J1), deepcopy(W1), deepcopy(B1)

        return J, W, B

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
        p, J, W, B = self.init()

        for i in range(N):
            progress(i, N, 'LARK')
            J, W, B = self.rj_mcmc(p, J, W, B)
            p = self.sample_p(p, W, B)
            # lam = self.sample_p(lam, p, W, B)
            if i > bip: res.append([p, J, W, B])

        self.res = res
        self.accept_pct = self.accepted/N*100
        print(f'\nAcceptence Ratio = {int(self.accept_pct)}%')
        return res

@timer
def plot_out(posterior, lark, pp=False):
    nu = lark.nu
    N = len(posterior)
    ps = []

    dom = linspace(0, 1, 1000)
    plt.plot(dom, Data.f(dom)**2, label='True volatility')

    plot_post = []
    for i, post in enumerate(posterior):
        progress(i, N, 'Plotting')
        p, J, W, B = post
        ps.append(p)
        plot_post.append([nu(x, p, W, B) for x in dom])
        if pp: plt.plot(dom, plot_post[-1], alpha=0.05, color='r')
    plot_post = matrix(plot_post)

    quantiles = []
    for i in range(1000): quantiles.append(quantile(plot_post[:,i], [0.025, 0.975]))
    quantiles = matrix(quantiles)

    plot_post = array(plot_post.mean(0))[0]
    plt.plot(dom, plot_post, label='Posterior Mean')
    plt.fill_between(dom, array(quantiles[:, 0].flatten())[0], array(quantiles[:, 1].flatten())[0], alpha=0.3, color='red')

    #plt.title(f'n={lark.n}, N={n}, kernel=$exp(-10|x-y|)$')
    plt.legend()
    plt.plot(lark.X, lark.Y, alpha=0.4, label='Observations', color='black')


    plt.figure()
    dom = linspace(0, max([3, max(ps)]), 1000)
    plt.plot(dom, [gamma.pdf(x, lark.ap, scale=1/lark.bp) for x in dom], label='prior')
    plt.hist(ps, bins=20, label='posterior', density=True)
    plt.legend()

    plt.figure()
    plt.plot([J for _, J, _, _ in posterior], label='J trace')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    parser.add_argument('--n', help='Sample size', type=int, default=100)
    parser.add_argument('--N', help='MCMC iterations', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period', type=int, default=0)
    parser.add_argument('--p', type=str, default='0.4,0.4,0.2')
    parser.add_argument('--post', help='Plot posterior samples', action='store_true')
    parser.add_argument('--plot', help='Plot output', action='store_true')
    parser.add_argument('--plot_samples', help='Plot output', action='store_true')
    parser.add_argument('--save', type=str, help='file name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='Kernel function', choices=['haar', 'expon'], default='expon')
    args = parser.parse_args()

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    X, Y = Data.gen_data(args.n)
    lark = LARK(X=X, Y=Y, p=p, kernel=args.kernel)
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
    if args.plot: plot_out(res, lark, args.plot_samples)

if __name__=='__main__':
    main()
