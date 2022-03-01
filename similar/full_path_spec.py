#!/usr/bin/env python3

from multiprocessing import Pool
import os
import argparse
import pandas as pd
import json
from time import time
import pdb
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import rand, randint, randn, exponential
from copy import deepcopy
from scipy.stats import poisson, gamma, norm, nbinom, chi2

from common import *
from kernels import Kernels
from getdata import Data

class LARK(Kernels):

    def __init__(self, T, X, p, eps, kernel, drift=None, **kwargs):
        '''
        kwargs are passed on as kernel parameters
        '''
        if not nomulti: self.pool = Pool(cores)
        
        self.ap = 2 # 4
        self.bp = 0.75 # 4

        self.al = 1.5
        self.bl = 10

        self.pb, self.pd, self.pu = p
        self.n = len(T)
        self.T, self.X = T, X
        self.cX = cumsum(X)
        self.b0 = 1e-8
        self.s_proposal = 0.1 # proposal std

        self.S = kernel
        self.dt = 1/self.n
        self.drift = drift
        model = 'dX = (a+bX)dt + s(X)dB' if drift else 'dX = s(X)dB'
        print(f'{model=} ')

        self.eps = eps
        self.birth = Birth(eps=eps, alpha=0.1, beta=1) # choose smarter a?
        self.wa, self.wb = 1.2*min(self.cX), 1.2*max(self.cX)

    def init_haar(self):
        J = poisson.rvs(10)
        self.J['haar'] = J
        self.s['haar'] = gamma.rvs(self.al, scale=1/self.bl)
        self.W['haar'] = list(rand(J)*(self.wb-self.wa)+self.wa)
        self.B['haar'] = list(self.birth.rvs(size=J))
    
    def init_expon(self):
        J = poisson.rvs(10)
        self.J['expon'] = J
        self.W['expon'] = list(rand(J)*(self.wb-self.wa)+self.wa)
        self.B['expon'] = list(self.birth.rvs(size=J))
        self.s['expon'] = gamma.rvs(self.al, scale=1/self.bl)
        self.p['expon'] = gamma.rvs(self.ap, scale=1/self.bp)

    def init_drift(self):
        self.a = norm.rvs(0, scale=1)
        self.b = gamma.rvs(self.al, scale=1/self.bl)

    def nu(self, t, p, s, W, B):
        out = self.b0
        if 'expon' in self.S:
            kernel = 'expon'
            out += sum([b*self.expon(t, w, p=p[kernel], s=s[kernel]) for w, b in zip(W[kernel], B[kernel])])

        if 'haar' in self.S:
            kernel = 'haar'
            out += sum([b*self.haar(t, w, s=s[kernel]) for w, b in zip(W[kernel], B[kernel])])

        return out


    def _l(self, i):
        t = self.T[i]
        x0, x1 = self.cX[i], self.cX[i+1]
        a, b, p, s, W, B = self.largs
        nui = self.nu(x0, p, s, W, B)
        mean = (a+b*x0)*self.dt
        std = sqrt(nui*self.dt)
        return norm.logpdf(x1-x0, loc=mean, scale=std)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def l(self, a=None, b=None, p=None, s=None, W=None, B=None):
        a = a if a else self.a
        b = b if b else self.b
        p = p if p else self.p
        s = s if s else self.s
        W = W if W else self.W
        B = B if B else self.B

        out = 0
        self.largs = [a, b, p, s, W, B]
        if nomulti:
            out = sum([self._l(i) for i in range(self.n-1)])
        else:
            iters = arange(self.n-1) # minus 1???
            out = sum(self.pool.map(self._l, iters))
        return out

    def rj_mcmc(self, kernel):
        J1 = deepcopy(self.J)
        W1 = deepcopy(self.W)
        B1 = deepcopy(self.B)
        u = rand()

        l0 = self.l()

        if u < self.pb or self.J[kernel] == 0: # birth
            J1[kernel] += 1
            w = rand()*(self.wb-self.wa)+self.wa
            b = self.birth.rvs()

            W1[kernel].append(w)
            B1[kernel].append(b)
            l1 = self.l(W=W1, B=B1)

            qd = norm.cdf((self.eps-b)/self.s_proposal)
            A1 = l1-l0
            #A2 = self.birth.logpdf(b)
            A3 = log(self.pd+self.pu*qd)+log(J1[kernel])
            A4 = -log(self.pb)-log(self.J[kernel])
            #A5 = -log(norm.logpdf(b, loc=b, s))
            #A6 = A1+A2+A3+A4+A4
            A6 = A1+A3+A4
            A = min([0, A6])
        else:
            j = randint(0, self.J[kernel])
            b = norm.rvs()*self.s_proposal+self.B[kernel][j]
            w = self.W[kernel][j]
            if b < self.eps or u < self.pb + self.pd: # death
                J1[kernel] -= 1
                j = randint(0, self.J[kernel])
                del W1[kernel][j], B1[kernel][j]
                
                l1 = self.l(W=W1, B=B1)

                qd = norm.cdf((self.eps-b)/self.s_proposal)
                A1 = l1-l0
                #A2 = -self.birth.logpdf(b)
                A3 = log(self.pb)+log(J1[kernel])
                A4 = -log(self.pd+self.pu*qd)-log(self.J[kernel])
                #A5 = lognorm(pdf(b[j))
                #A6 = A1+A2+A3+A4+A5
                A6 = A1+A3+A4
                A = min([0, A6])
            else: # update
                bold = B1[kernel][j]
                W1[kernel][j] = w
                B1[kernel][j] = b

                l1 = self.l(W=W1, B=B1)

                A1 = l1-l0
                A2 = self.birth.logpdf(b)-self.birth.logpdf(bold)
                A3 = norm.logpdf(b, loc=bold, scale=self.s_proposal)
                A4 = -norm.logpdf(bold, loc=b, scale=self.s_proposal)
                A5 = A1+A2+A3+A4
                A = min([0, A5])

        e = exponential(1)
        if e+A > 0:
            self.accepted[kernel] += 1
            self.J, self.W, self.B = deepcopy(J1), deepcopy(W1), deepcopy(B1)

    def sample_s(self, kernel):
        s1 = deepcopy(self.s)
        s1[kernel] = gamma.rvs(self.al, scale=1/self.bl)

        l0 = self.l()
        l1 = self.l(s=s1)
        A1 = l1-l0+gamma.logpdf(self.s[kernel], self.al, scale=1/self.bl)-gamma.logpdf(s1[kernel], self.al, scale=1/self.bl)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.s = s1
            self.accepted['s'] += 1

    def sample_p(self, kernel):
        p1 = deepcopy(self.p)
        p1[kernel] = gamma.rvs(self.ap, scale=1/self.bp)

        l0 = self.l()
        l1 = self.l(p=p1)
        A1 = l1-l0+gamma.logpdf(self.p[kernel], self.ap, scale=1/self.bp)-gamma.logpdf(p1[kernel], self.ap, scale=1/self.bp)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.p = p1
            self.accepted['p'] += 1

    def sample_a(self):
        a1 = norm.rvs(0, scale=1)

        l0 = self.l()
        l1 = self.l(a=a1)
        A1 = l1-l0+norm.logpdf(self.a, 0, scale=1)-norm.logpdf(a1, 0, scale=1)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.a = a1
            self.accepted['a'] += 1

    def sample_b(self):
        b1 = gamma.rvs(self.al, scale=1/self.bl)

        l0 = self.l()
        l1 = self.l(b=b1)
        A1 = l1-l0+gamma.logpdf(self.b, self.al, scale=1/self.bl)-gamma.logpdf(b1, self.al, scale=1/self.bl)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.b = b1
            self.accepted['b'] += 1

    def save(self, fn):
        out = {
            'T': list(self.T),
            'X': list(self.X),
            'post': list(self.res)
            }
        with open(fn, 'w') as f:
            json.dump(out, f)

    @timer
    def __call__(self, N=100, bip=0):
        self.accepted = {'expon': 0, 'haar': 0, 'p': 0, 's': 0, 'a': 0, 'b': 0}
        res = []

        self.p, self.s = {'expon': 0}, {'expon': 0, 'haar': 0}
        self.J, self.W, self.B = {}, {}, {}
        self.a, self.b = 0, 0

        if 'expon' in self.S: 
            self.init_expon()

        if 'haar' in self.S: 
            self.init_haar()

        if self.drift:
            self.init_drift()

        for i in range(N):
            progress(i, N, 'LARK')

            if 'expon' in self.J:
                self.rj_mcmc('expon')
                self.sample_s('expon')
                self.sample_p('expon')

            if 'haar' in self.J:
                self.rj_mcmc('haar')
                self.sample_s('haar')

            if self.drift:
                self.sample_a()
                self.sample_b()

            if i > bip: res.append([self.a, self.b, self.p, self.s, self.J, self.W, self.B])

        self.res = res
        for k, v in self.accepted.items():
            accept_pct = v/N*100
            print(f'\nAcceptence [{k}] = {int(accept_pct)}%')
        return res

@timer
def plot_out(posterior, lark, pp=False, real=False, mcmc_res=False):
    m = 1000
    nu = lark.nu
    N = len(posterior)
    ps, ss = [], []
    aa, bs = [], []

    dom = linspace(min(lark.cX), max(lark.cX), m)

    if not real: 
        plt.plot(dom, Data.sigx(dom)**2, label='True volatility')

    plot_post = []
    for i, post in enumerate(posterior):
        progress(i, N, 'Plotting')
        a, b, p, s, J, W, B = post
        ps.append(p)
        ss.append(s)
        aa.append(a)
        bs.append(b)
        plot_post.append([nu(x, p, s, W, B) for x in dom])
        if pp: plt.plot(dom, plot_post[-1], alpha=0.05, color='r')
    plot_post = matrix(plot_post)

    quantiles = []
    for i in range(m): quantiles.append(quantile(plot_post[:,i], [0.025, 0.975]))
    quantiles = matrix(quantiles)

    plot_post = array(plot_post.mean(0))[0]
    plt.plot(dom, plot_post, label='Posterior Mean')
    plt.fill_between(dom, array(quantiles[:, 0].flatten())[0], array(quantiles[:, 1].flatten())[0], alpha=0.3, color='red')

    plt.legend()

    if mcmc_res:
        plt.figure()
        dom = linspace(0, max([3, max([p['expon'] for p in ps])]), m)
        #dom = linspace(0, max([3, max(ps['expon'])]), m)
        plt.plot(dom, [gamma.pdf(x, lark.ap, scale=1/lark.bp) for x in dom], label='prior')
        plt.hist([p['expon'] for p in ps], bins=20, label='posterior', density=True)
        plt.title('p')
        plt.legend()

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(lark.T, lark.X, alpha=0.4, label='Observations', color='black')
        ax[1].plot(lark.T, lark.cX, alpha=0.4, label='Observations', color='black')

        plt.figure()
        dom = linspace(0, 0.5, m)
        plt.plot(dom, [gamma.pdf(x, lark.al, scale=1/lark.bl) for x in dom], label='prior')
        if 'expon' in lark.S: plt.hist([x['expon'] for x in ss], bins=20, label='Posterior expon', density=True, alpha=0.8)
        if 'haar' in lark.S: plt.hist([x['haar'] for x in ss], bins=20, label='Posterior haar', density=True, alpha=0.8)
        plt.title('s')
        plt.legend()

        plt.figure()
        if 'expon' in lark.S: plt.plot([J['expon'] for _, _, _, _, J, _, _ in posterior], label='J expon trace')
        if 'haar' in lark.S: plt.plot([J['haar'] for _, _, _, _, J, _, _ in posterior], label='J haar trace')
        if lark.drift:
            plt.figure()
            dom = linspace(-3, 3, m)
            plt.plot(dom, [norm.pdf(x, 0, scale=1) for x in dom], label='prior')
            plt.hist(aa, bins=20, label='Posterior', density=True, alpha=0.8)
            plt.legend()
            plt.title('a')

            plt.figure()
            dom = linspace(0, 3, m)
            plt.plot(dom, [gamma.pdf(x, lark.al, scale=1/lark.bl) for x in dom], label='prior')
            plt.hist(bs, bins=20, label='Posterior', density=True, alpha=0.8)
            plt.legend()
            plt.title('b')
        plt.legend()
    plt.show()

def main():
    global nomulti, cores
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    parser.add_argument('--n', help='Sample size [100]', type=int, default=100)
    parser.add_argument('--N', help='MCMC iterations [1000]', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period [0]', type=int, default=0)
    parser.add_argument('--eps', help='epsilon [0.5]', type=float, default=0.5)
    parser.add_argument('--p', type=str, default='0.4,0.4,0.2')
    parser.add_argument('--drift', help='Use drift of the form a+bX', action='store_true')
    parser.add_argument('--real', help='Use real data', action='store_true')
    parser.add_argument('--ticker', type=str, help='ticker to get data', default='AAPL')
    parser.add_argument('--noplot', help='Plot output', action='store_true')
    parser.add_argument('--nomulti', help='no multiprocessing', action='store_true')
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=os.cpu_count())
    parser.add_argument('--no_mcmc_plot', help='don\'t show mcmc convergence pltos', action='store_true')
    parser.add_argument('--plot_samples', help='Plot output', action='store_true')
    parser.add_argument('--save', type=str, help='file name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='comma separated kernel functions', default='expon,haar')
    args = parser.parse_args()

    nomulti = args.nomulti
    cores = args.cores

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    if args.real:
        T, X, dB = Data.get_stock(n=args.n, ticker=args.ticker, returns=False)
    else:
        T, X, dB = Data.gen_data_x(n=args.n)

    lark = LARK(T=T, X=X, p=p, eps=args.eps, kernel=args.kernel.split(','), drift=args.drift)
    if not args.load:
        res = lark(N=args.N, bip=args.bip)
    else:
        with open(args.load) as fn:
            data = json.load(fn)
        res = data['post']
        lark.T = data['T']
        lark.X = data['X']
        lark.res = res

    if args.save: lark.save(args.save)
    if not args.noplot: plot_out(res, lark, args.plot_samples, real=args.real, mcmc_res=not args.no_mcmc_plot)

if __name__=='__main__':
    main()
