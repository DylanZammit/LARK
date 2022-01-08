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
from scipy.stats import rv_continuous
from scipy.special import exp1

from common import *
from kernels import Kernels
from getdata import Data

def expinv(x, d=1e-9):
    return chi2.ppf(1-x*d/2, d)/2

class Birth(rv_continuous):

    def __init__(self, eps, alpha, beta, *args, **kwargs):
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        assert alpha > 0 and beta > 0
        super().__init__(*args, **kwargs)

    def _pdf(self, x):
        return self.alpha*exp(-self.beta*x)/x/exp1(self.eps)*(x>=self.eps)

    def _logpdf(self, x):
        if x < self.eps: 
            import pdb; pdb.set_trace()
            raise Exception
        return log(self.alpha)-self.beta*x

    def _ppf(self, x):
        return expinv(exp1(self.eps)*(1-x))

class LARK(Kernels):

    def __init__(self, X, Y, p, eps, kernel, drift=None, **kwargs):
        '''
        kwargs are passed on as kernel parameters
        '''
        if not nomulti: self.pool = Pool(cores)
        
        self.a = {'expon': 1, 'haar': 1}
        self.b = {'expon': 1, 'haar': 1}

        #self.a = {'expon': 5, 'haar': 0.01}
        #self.b = {'expon': 2, 'haar': 1}

        self.ap = 2 # 4
        self.bp = 0.75 # 4

        self.al = 1.5
        self.bl = 10

        self.pb, self.pd, self.pu = p
        self.n = len(X)
        self.X, self.Y = X, Y
        self.b0 = 1e-8
        self.s = 0.1 # proposal std

        self.S = kernel
        self.dt = 1/self.n
        if drift == 'linear':
            self.mu = {x: mean(Y) for x in X}
        elif drift == 'zero':
            self.mu = {x: 0 for x in X}
        elif drift.startswith('EMA'): # ex EMA20
            w = int(drift[3:])
            mu = pd.Series(Y).ewm(w).mean().values
            self.mu = {x: m for x, m in zip(X, mu)}
        print(f'{drift=} ')

        self.eps = eps
        self.birth = Birth(eps=eps, alpha=0.1, beta=1) # choose smarter a?

    def init_haar(self):
        J = poisson.rvs(10)
        W = rand(J)
        B = gamma.rvs(self.a['haar'], scale=1/self.b['haar'], size=J)
        s = gamma.rvs(self.al, scale=1/self.bl)
        return s, J, list(W), list(B)
    
    def init_expon(self):
        J = poisson.rvs(10)
        W = rand(J)
        B = self.birth.rvs(size=J)
        p = gamma.rvs(self.ap, scale=1/self.bp)
        s = gamma.rvs(self.al, scale=1/self.bl)
        return p, s, J, list(W), list(B)

    def nu(self, x, p, s, W, B):
        T = self.b0
        if 'expon' in self.S:
            kernel = 'expon'
            T += sum([b*self.expon(x, w, p=p, s=s[kernel]) for w, b in zip(W[kernel], B[kernel])])
            #T += sum([b*self.expon_asym2(x, w, p=p, s=s[kernel]) for w, b in zip(W[kernel], B[kernel])])

        if 'haar' in self.S:
            kernel = 'haar'
            T += sum([b*self.haar(x, w, s=s[kernel]) for w, b in zip(W[kernel], B[kernel])])

        return T


    def _l(self, i):
        t = self.X[i]
        x = self.Y[i]
        p, s, W, B = self.largs
        nui = self.nu(t, p, s, W, B)
        mean = self.mu[t]*self.dt
        std = sqrt(nui*self.dt)
        return norm.logpdf(x, loc=mean, scale=std)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def l(self, p, s, W, B):
        out = 0
        if nomulti:
            for t, x in zip(self.X, self.Y):
                nui = self.nu(t, p, s, W, B)
                mean = self.mu[t]*self.dt
                std = sqrt(nui*self.dt)
                out += norm.logpdf(x, loc=mean, scale=std)
            return out
        else:
            self.largs = [p, s, W, B]
            iters = arange(self.n)
            X = self.pool.map(self._l, iters)
            return sum(X)

    def rj_mcmc(self, p, s, J, W, B, kernel):
        p1 = p
        s1 = deepcopy(s)
        J1 = deepcopy(J)
        W1 = deepcopy(W)
        B1 = deepcopy(B)
        u = rand()

        l0 = self.l(p, s, W, B)

        if u < self.pb or J[kernel] == 0: # birth
            J1[kernel] += 1
            w = rand()
            b = self.birth.rvs()

            W1[kernel].append(w)
            B1[kernel].append(b)
            l1 = self.l(p1, s1, W1, B1)

            qd = norm.cdf((self.eps-b)/self.s)
            A1 = l1-l0
            #A2 = self.birth.logpdf(b)
            A3 = log(self.pd+self.pu*qd)+log(J1[kernel])
            A4 = -log(self.pb)-log(J[kernel])
            #A5 = -log(norm.logpdf(b, loc=b, s))
            #A6 = A1+A2+A3+A4+A4
            A6 = A1+A3+A4
            A = min([0, A6])
        else:
            j = randint(0, J[kernel])
            b = norm.rvs()*self.s+B[kernel][j]
            w = W[kernel][j]
            if b < self.eps or u < self.pb + self.pd: # death
                J1[kernel] -= 1
                j = randint(0, J[kernel])
                del W1[kernel][j], B1[kernel][j]
                
                l1 = self.l(p1, s1, W1, B1)

                qd = norm.cdf((self.eps-b)/self.s)
                A1 = l1-l0
                #A2 = -self.birth.logpdf(b)
                A3 = log(self.pb)+log(J1[kernel])
                A4 = -log(self.pd+self.pu*qd)-log(J[kernel])
                #A5 = lognorm(pdf(b[j))
                #A6 = A1+A2+A3+A4+A5
                A6 = A1+A3+A4
                A = min([0, A6])
            else: # update
                bold = B1[kernel][j]
                W1[kernel][j] = w
                B1[kernel][j] = b

                l1 = self.l(p1, s1, W1, B1)

                A1 = l1-l0
                A2 = self.birth.logpdf(b)-self.birth.logpdf(bold)
                A3 = norm.logpdf(b, loc=bold, scale=self.s)
                A4 = -norm.logpdf(bold, loc=b, scale=self.s)
                A5 = A1+A2+A3+A4
                A = min([0, A5])

        e = exponential(1)
        if e+A > 0:
            self.accepted[kernel] += 1
            J, W, B = deepcopy(J1), deepcopy(W1), deepcopy(B1)

        return J, W, B

    def sample_s(self, p, s, W, B, kernel):
        s1 = deepcopy(s)
        s1[kernel] = gamma.rvs(self.al, scale=1/self.bl)
        #print('\n', s, s1)

        l0 = self.l(p, s, W, B)
        l1 = self.l(p, s1, W, B)
        A1 = l1-l0+gamma.logpdf(s[kernel], self.al, scale=1/self.bl)-gamma.logpdf(s1[kernel], self.al, scale=1/self.bl)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            s = s1
            self.accepted['s'] += 1
        return s

    def sample_p(self, p, s, W, B, kernel):
        p1 = gamma.rvs(self.ap, scale=1/self.bp)

        l0 = self.l(p, s, W, B)
        l1 = self.l(p1, s, W, B)
        A1 = l1-l0+gamma.logpdf(p, self.ap, scale=1/self.bp)-gamma.logpdf(p1, self.ap, scale=1/self.bp)
        A = min([0, A1])

        e = exponential(1)
        if e+A > 0: 
            self.accepted['p'] += 1
            p = p1
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
        self.accepted = {'expon': 0, 'haar': 0, 'p': 0, 's': 0}
        res = []

        s, J, W, B = {'expon': 0, 'haar': 0}, {}, {}, {}
        p = 0# in case not defined

        if 'expon' in self.S: 
            p, se, Je, We, Be = self.init_expon()
            s['expon'] = se
            J['expon'] = Je
            W['expon'] = We
            B['expon'] = Be
        if 'haar' in self.S: 
            sh, Jh, Wh, Bh = self.init_haar()
            s['haar'] = sh
            J['haar'] = Jh
            W['haar'] = Wh
            B['haar'] = Bh

        for i in range(N):
            progress(i, N, 'LARK')

            if 'expon' in J:
                J, W, B = self.rj_mcmc(p, s, J, W, B, 'expon')
                s = self.sample_s(p, s, W, B, 'expon')
                p = self.sample_p(p, s, W, B, 'expon')

            if 'haar' in J:
                J, W, B = self.rj_mcmc(p, s, J, W, B, 'haar')
                s = self.sample_s(p, s, W, B, 'haar')

            if i > bip: res.append([p, s, J, W, B])

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

    dom = linspace(0, 1, m)

    if not real: 
        plt.plot(dom, Data.sigt(dom)**2, label='True volatility')
    else:
        import pandas as pd
        # multiply by lark.n??
        rollvar = pd.Series(lark.Y).ewm(10).var().bfill().values*lark.n
        plt.plot(linspace(0, 1, lark.n), rollvar, label='rolling var')

    plot_post = []
    for i, post in enumerate(posterior):
        progress(i, N, 'Plotting')
        p, s, J, W, B = post
        ps.append(p)
        ss.append(s)
        plot_post.append([nu(x, p, s, W, B) for x in dom])
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
    plt.plot(lark.X, lark.Y, alpha=0.4, label='Observations', color='black')


    if mcmc_res:
        plt.figure()
        dom = linspace(0, max([3, max(ps)]), m)
        plt.plot(dom, [gamma.pdf(x, lark.ap, scale=1/lark.bp) for x in dom], label='prior')
        plt.hist(ps, bins=20, label='posterior', density=True)
        plt.title('p')
        plt.legend()

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(lark.X, lark.Y, alpha=0.4, label='Observations', color='black')
        ax[1].plot(lark.X, cumsum(lark.Y), alpha=0.4, label='Observations', color='black')

        plt.figure()
        dom = linspace(0, 0.5, m)
        plt.plot(dom, [gamma.pdf(x, lark.al, scale=1/lark.bl) for x in dom], label='prior')
        if 'expon' in lark.S: plt.hist([x['expon'] for x in ss], bins=20, label='Posterior expon', density=True, alpha=0.8)
        if 'haar' in lark.S: plt.hist([x['haar'] for x in ss], bins=20, label='Posterior haar', density=True, alpha=0.8)
        plt.title('s')
        plt.legend()

        plt.figure()
        if 'expon' in lark.S: plt.plot([J['expon'] for _, _, J, _, _ in posterior], label='J expon trace')
        if 'haar' in lark.S: plt.plot([J['haar'] for _, _, J, _, _ in posterior], label='J haar trace')
        plt.legend()
    plt.show()

def main():
    global nomulti, cores
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    parser.add_argument('--n', help='Sample size', type=int, default=100)
    parser.add_argument('--N', help='MCMC iterations', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period', type=int, default=0)
    parser.add_argument('--eps', help='epsilon', type=float, default=0.5)
    parser.add_argument('--p', type=str, default='0.4,0.4,0.2')
    parser.add_argument('--drift', type=str, default='zero')
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
        X, Y, dB = Data.get_stock(n=args.n, ticker=args.ticker)
    else:
        X, Y, dB = Data.gen_data_t(n=args.n)

    lark = LARK(X=X, Y=Y, p=p, eps=args.eps, kernel=args.kernel.split(','), drift=args.drift)
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
    if not args.noplot: plot_out(res, lark, args.plot_samples, real=args.real, mcmc_res=not args.no_mcmc_plot)

if __name__=='__main__':
    main()
