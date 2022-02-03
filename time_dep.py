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
        
        self.ap = 4
        self.bp = 2

        self.al = 2
        self.bl = 200

        self.pb, self.pd, self.pu = p
        self.n = len(T)
        self.T, self.X = T, X
        self.b0 = 1e-8
        self.b_proposal = 0.1 # depending on application
        self.w_proposal = 0.1
        self.s_proposal = 0.2
        self.p_proposal = 0.2

        self.kernels = kernel
        self.dt = 1/self.n # assume domain is [0,1]
        if drift == 'linear':
            self.mu = {t: mean(X) for t in T}
        elif drift == 'zero':
            self.mu = {t: 0 for t in T}
        elif drift.startswith('EMA'): # ex EMA20
            w = int(drift[3:])
            mu = pd.Series(X).ewm(w).mean().values
            self.mu = {t: m for t, m in zip(T, mu)}
        print(f'{drift=} ')

        self.eps = eps
        #alpha = 1
        alpha = 1
        beta = 1
        self.birth = Birth(eps=eps, alpha=alpha, beta=beta) # can model a
        self.vplus = 30

    def init(self, kernel):
        J = poisson.rvs(self.vplus)
        self.J[kernel] = J
        self.W[kernel] = list(rand(J))
        self.B[kernel] = list(self.birth.rvs(size=J))
        self.S[kernel] = list(gamma.rvs(self.al, scale=1/self.bl, size=J))
        self.P[kernel] = list(gamma.rvs(self.ap, scale=1/self.bp, size=J))
    
    def nu(self, t, P, S, W, B):
        out = self.b0
        for kernel in self.kernels:
            k = getattr(self, kernel)
            out += sum([b*k(t, w, p=p, s=s) for w, b, p, s in zip(W[kernel], B[kernel], P[kernel], S[kernel])])
        return out

    def _l(self, i):
        t = self.T[i]
        x = self.X[i]
        P, S, W, B = self.largs
        nui = self.nu(t, P, S, W, B)
        mean = self.mu[t]*self.dt
        std = sqrt(nui*self.dt)
        return norm.logpdf(x, loc=mean, scale=std)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def l(self, P=None, S=None, W=None, B=None):
        P = P if P else self.P
        S = S if S else self.S
        W = W if W else self.W
        B = B if B else self.B

        out = 0
        self.largs = [P, S, W, B]
        if nomulti:
            out = sum([self._l(i) for i in range(self.n)])
        else:
            iters = arange(self.n)
            out = sum(self.pool.map(self._l, iters))
        return out

    def rj_mcmc(self, kernel):
        J1 = deepcopy(self.J)
        W1 = deepcopy(self.W)
        B1 = deepcopy(self.B)
        S1 = deepcopy(self.S)
        P1 = deepcopy(self.P)
        u = rand()

        l0 = self.l()
        self.update_step = False

        if u < self.pb or self.J[kernel] == 0: # birth
            J1[kernel] += 1
            w = rand()
            b = self.birth.rvs()
            s = gamma.rvs(self.al, scale=1/self.bl)
            p = gamma.rvs(self.ap, scale=1/self.bp)

            W1[kernel].append(w)
            B1[kernel].append(b)
            S1[kernel].append(s)
            P1[kernel].append(p)
            l1 = self.l(W=W1, B=B1, S=S1, P=P1)

            qd = norm.cdf((self.eps-b)/self.b_proposal)
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
            b = norm.rvs()*self.b_proposal+self.B[kernel][j]
            if b < self.eps or u < self.pb + self.pd: # death
                J1[kernel] -= 1
                del W1[kernel][j], B1[kernel][j], S1[kernel][j], P1[kernel][j]
                
                l1 = self.l(W=W1, B=B1, S=S1, P=P1)

                qd = norm.cdf((self.eps-b)/self.b_proposal)
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
                B1[kernel][j] = b

                l1 = self.l(W=W1, B=B1)

                A1 = l1-l0
                A2 = self.birth.logpdf(b)-self.birth.logpdf(bold)
                A3 = norm.logpdf(b, loc=bold, scale=self.b_proposal)
                A4 = -norm.logpdf(bold, loc=b, scale=self.b_proposal)
                A5 = A1+A2+A3+A4
                A = min([0, A5])

                self.update_step = True
                self.update_comp = j

        e = exponential(1)
        if e+A > 0:
            self.accepted[kernel] += 1
            self.J, self.W, self.B, self.S, self.P = deepcopy(J1), deepcopy(W1), deepcopy(B1), deepcopy(S1), deepcopy(P1)
        else:
            self.update_step = False

    def sample_s(self, kernel):
        S1 = deepcopy(self.S)
        sold = log(S1[kernel][self.update_comp])
        snew = sold + norm.rvs()*self.s_proposal
        S1[kernel][self.update_comp] = exp(snew)

        l0 = self.l()
        l1 = self.l(S=S1)
        A1 = l1-l0
        A2 = gamma.logpdf(exp(sold), self.al, scale=1/self.bl)
        A3 = -gamma.logpdf(exp(snew), self.al, scale=1/self.bl)
        A4 = A1+A2+A3
        A = min([0, A4])

        e = exponential(1)
        if e+A > 0: 
            self.S = deepcopy(S1)
            self.accepted['s'] += 1

    def sample_p(self, kernel):
        P1 = deepcopy(self.P)
        pold = log(P1[kernel][self.update_comp])
        pnew = pold + norm.rvs()*self.p_proposal
        P1[kernel][self.update_comp] = exp(pnew)

        l0 = self.l()
        l1 = self.l(P=P1)
        A1 = l1-l0
        A2 = gamma.logpdf(exp(pold), self.ap, scale=1/self.bp)
        A3 = -gamma.logpdf(exp(pnew), self.ap, scale=1/self.bp)
        A4 = A1+A2+A3
        A = min([0, A4])

        e = exponential(1)
        if e+A > 0: 
            self.P = P1
            self.accepted['p'] += 1

    def sample_w(self, kernel):
        W1 = deepcopy(self.W)
        wold = W1[kernel][self.update_comp]
        wnew = wold + norm.rvs()*self.w_proposal
        if not 0 < wnew < 1: return

        W1[kernel][self.update_comp] = exp(wnew)

        l0 = self.l()
        l1 = self.l(W=W1)
        A1 = l1-l0
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.W = W1
            self.accepted['w'] += 1

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
        self.accepted = {k: 0 for k in self.kernels}
        self.accepted.update({k: 0 for k in ['s','p', 'w']})

        res = []

        self.P, self.S, self.J, self.W, self.B = {}, {}, {}, {}, {}

        for k in self.kernels: self.init(k)

        for i in range(N):
            progress(i, N, 'LARK')

            #loop through chosen kernels
            for k in self.kernels:
                self.rj_mcmc(k)
                if self.update_step:
                    self.sample_p(k)
                    self.sample_s(k)
                    self.sample_w(k)

            if i > bip: res.append([self.P, self.S, self.J, self.W, self.B])

        self.res = res
        for k, v in self.accepted.items():
            accept_pct = v/N*100
            print(f'\nAcceptence [{k}] = {int(accept_pct)}%')
        return res

@timer
def plot_out(posterior, lark, pp=False, real=False):
    m = 1000
    nu = lark.nu
    N = len(posterior)
    ps, ss = [], []

    dom = linspace(0, 1, m)

    if not real: 
        plt.plot(dom, Data.sigt(dom)**2, label='True volatility')
    else:
        import pandas as pd
        # multiply by lark.n??
        rollvar = pd.Series(lark.X).ewm(10).var().bfill().values*lark.n
        plt.plot(linspace(0, 1, lark.n), rollvar, label='rolling var')

    plot_post = []
    for i, post in enumerate(posterior):
        progress(i, N, 'Plotting')
        P, S, J, W, B = post
        #ps.append(p)
        #ss.append(s)
        plot_post.append([nu(x, P, S, W, B) for x in dom])
        if pp: plt.plot(dom, plot_post[-1], alpha=0.05, color='r')
    plot_post = matrix(plot_post)

    quantiles = []
    for i in range(m): quantiles.append(quantile(plot_post[:,i], [0.025, 0.975]))
    quantiles = matrix(quantiles)

    plot_post = array(plot_post.mean(0))[0]
    plt.plot(dom, plot_post, label='Posterior Mean')
    plt.fill_between(dom, array(quantiles[:, 0].flatten())[0], array(quantiles[:, 1].flatten())[0], alpha=0.3, color='red')

    #plt.title(f'n={lark.n}, N={n}, kernel=$exp(-10|x-y|)$')
    plt.legend()
    plt.plot(lark.T, lark.X, alpha=0.4, label='Observations', color='black')

    plt.figure()
    for k in lark.kernels:
        plt.plot([J[k] for _, _, J, _, _ in posterior], label=f'J {k} trace')

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
    parser.add_argument('--drift', type=str, default='zero')
    parser.add_argument('--real', help='Use real data', action='store_true')
    parser.add_argument('--ticker', type=str, help='ticker to get data', default='AAPL')
    parser.add_argument('--noplot', help='Plot output', action='store_true')
    parser.add_argument('--nomulti', help='no multiprocessing', action='store_true')
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=os.cpu_count())
    #parser.add_argument('--no_mcmc_plot', help='don\'t show mcmc convergence pltos', action='store_true')
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
        T, X, dB = Data.get_stock(n=args.n, ticker=args.ticker)
    else:
        T, X, dB = Data.gen_data_t(n=args.n)

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
    if not args.noplot: plot_out(res, lark, args.plot_samples, real=args.real)

if __name__=='__main__':
    main()
