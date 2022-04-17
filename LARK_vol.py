#!/usr/bin/env python3

import yaml
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
from scipy.stats import poisson, gamma, norm, nbinom, chi2, lognorm, pareto

from common import *
from kernels import Kernels
from getdata import Data
import time
import warnings
warnings.filterwarnings("ignore")

data = '/home/dylan/git/LARK/data'
START = time.time()
LASTLOG = 0
profile = False

timers = {}
def time_avg(f):
    def wrap(*args, **kwargs):
        global LASTLOG
        start = time.time()
        result = f(*args, **kwargs)
        if not profile: return result
        end = time.time()
        this_time= end-start
        if f.__name__ not in timers:
            n = 0
            avg_time = this_time
            tot_time = this_time
        else:
            avg_time, tot_time, n = timers[f.__name__]
            avg_time = (avg_time*n+this_time)/(n+1)
            tot_time += this_time
        timers[f.__name__] = (avg_time, tot_time, n+1)
        prog_time = int(time.time()-START)
        if prog_time-LASTLOG>=5:
            LASTLOG = prog_time
            prog_time = str(prog_time)+'s'
            stimers = sorted(timers.items(), key=lambda x:x[1][1], reverse=True)
            print('-'*30, f'{prog_time:<8s}', '-'*30)
            print('{:<20s}{:<20s}{:<20s}{:<20s}'.format('Function', 'TotTime(s)', 'AvgTime(ms)', 'n'))
            print('-'*70)
            for k, v in stimers:
                print('{:<20s}{:<20f}{:<20f}{:<20d}'.format(k, round(v[1], 4), round(v[0]*100, 4), v[2]))
            m = len(stimers)+4
            print('\033[F')
            print('\033[A'*m)
        return result
    return wrap

class LARK(Kernels):

    def __init__(self, T, X, p, eps, kernel, drift=None, nomulti=False, cores=4, 
                 nu=1, gammap=None, gammal=None, proposals=None, vplus=10, stable=False, **kwargs):
        '''
        kwargs are passed on as kernel parameters
        '''
        if not nomulti: self.pool = Pool(cores)
        self.nomulti=nomulti
        
        self.ap, self.bp = gammap

        self.al, self.bl = gammal

        self.b_proposal = proposals[0]
        self.w_proposal = proposals[1]
        self.s_proposal = proposals[2]
        self.p_proposal = proposals[3]

        self.pb, self.pd, self.pu = p
        self.vplus = vplus

        self.n = len(T)
        self.T, self.X = T, X

        self.b0 = quantile(abs(self.X), 0.01)/2

        self.kernel = getattr(Kernels(), kernel)

        self.dt = 1
        if drift == 'linear':
            self.mu = {t: mean(X) for t in T}
        elif drift == 'zero':
            self.mu = {t: 0 for t in T}
        elif drift.startswith('EMA'): # ex EMA20
            w = int(drift[3:])
            mu = pd.Series(X).ewm(w).mean().values
            self.mu = {t: m for t, m in zip(T, mu)}
        print(f'{drift=} ')

        if not stable:
            self.birth = Gamma(eps=eps/nu, nu=nu)
            self.eps = eps/nu
        else:
            alpha = kwargs.get('alpha')
            self.birth = Pareto(eps=eps, nu=nu, alpha=alpha)
            #self.birth = SaS(eps=eps, alpha=nu)
            self.eps = eps/nu
        self.stable = stable
        print('eps={}, nu={}, vplus={}'.format(eps, nu, vplus))
        print('ap={}, bp={}, al={}, bl={}'.format(self.ap, self.bp, self.al, self.bl))

        self.l0 = None
        self.cache_l0 = True

    @time_avg
    def init(self):
        J = poisson.rvs(self.vplus)
        self.J = J
        self.W = list(rand(J))
        self.B = list(self.birth.rvs(size=J))
        self.S = list(gamma.rvs(self.al, scale=1/self.bl, size=J))
        self.p = gamma.rvs(self.ap, scale=1/self.bp)
    
    def nu(self, t, p, S, W, B):
        k = self.kernel
        out = self.b0 + B@k(t, y=W, p=p, s=S)
        #out = self.b0 + sum([b*k(t, y=w, p=p, s=s) for b, w, s in zip(B, W, S)])
        return out

    def _l(self, i):
        t = self.T[i]
        x = self.X[i]
        p, S, W, B = self.largs
        nui = self.nu(t, p, S, W, B)
        mean = self.mu[t]*self.dt
        std = sqrt(nui*self.dt)

        return norm.logpdf(x, loc=mean, scale=std)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @time_avg
    def l(self, p=None, S=None, W=None, B=None):
        if p is None and S is None and W is None and B is None and self.l0 is not None and self.cache_l0:
            return self.l0
        p = p if p is not None else self.p
        S = S if S is not None else self.S
        W = W if W is not None else self.W
        B = B if B is not None else self.B

        self.largs = [p, S, W, B]
        iters = arange(self.n)
        if self.nomulti:
            out = sum([self._l(i) for i in iters])
        else:
            out = sum(self.pool.map(self._l, iters))
        return out

    @time_avg
    def qb(self, l0, l1, bnew):
        qd = norm.cdf((self.eps-bnew)/self.b_proposal)
        l = l1-l0
        prior = log(self.vplus)-log(self.J+1)
        prop_n = log(self.pd+self.pu*qd)-log(self.J+1)
        prop_d = log(self.pb)
        prop = prop_n-prop_d

        return min([0, l+prior+prop])

    @time_avg
    def qd(self, l0, l1, bold):
        qd = norm.cdf((self.eps-bold)/self.b_proposal)
        l = l1-l0
        prior = log(self.J)-log(self.vplus)
        prop_n = log(self.pb)
        prop_d = log(self.pd+self.pu*qd)-log(self.J)
        prop = prop_n-prop_d

        return min([0, l+prior+prop])

    @time_avg
    def qu(self, l0, l1, bold, bnew):
        l = l1-l0
        prior = self.birth.logpdf(bnew)-self.birth.logpdf(bold)
        prop = 0

        return min([0, l+prior+prop])

    @time_avg
    def rj_mcmc(self):
        J1 = deepcopy(self.J)
        W1 = deepcopy(self.W)
        B1 = deepcopy(self.B)
        S1 = deepcopy(self.S)
        u = rand()

        l0 = self.l()
        birth, death, update = False, False, False

        if u < self.pb or self.J == 0: # birth
            birth = True
            J1 += 1
            w = rand()*1.1-0.1
            b = self.birth.rvs()
            s = gamma.rvs(self.al, scale=1/self.bl) 

            W1.append(w)
            B1.append(b)
            S1.append(s)
            l1 = self.l(W=W1, B=B1, S=S1)

            q = self.qb(l0, l1, b)

        else:
            j = randint(0, self.J)
            b = norm.rvs()*self.b_proposal+self.B[j]
            if b < self.eps or u < self.pb + self.pd: # death
                death = True
                J1 -= 1
                del W1[j], B1[j], S1[j]
                
                l1 = self.l(W=W1, B=B1, S=S1)
                bold = self.B[j]
                q = self.qd(l0, l1, bold)

            else: # update
                update = True
                bold = B1[j]
                B1[j] = b

                l1 = self.l(B=B1)
                q = self.qu(l0, l1, bold, b)

        self.update = update
        if update: self.update_comp = j

        e = exponential(1)
        if e+q > 0:
            accepted = True
            self.accepted['K'] += 1
            self.J, self.W, self.B, self.S = deepcopy(J1), deepcopy(W1), deepcopy(B1), deepcopy(S1)
            self.l0 = l1
        else:
            if self.l0 is None: self.l0 = l0
            accepted = False

        if 0: print('B, D, U={}, {}, {}....accepted={}'.format(birth, death, update, accepted))
        ######FOR DEBUG########
        #if (self.iter > 200 ) and 1: 
        if 0 and (self.iter % 30 == 0): 
        #if self.iter > 100 and death and not accepted:
            def myplot():
                dom = linspace(0, 1, 100)
                plt.title(f'B, D, U = {birth}, {death}, {update}')
                plt.plot(self.T, self.X, color='black', alpha=0.3)
                plt.plot(dom, [sqrt(self.nu(t, self.p, self.S, self.W, self.B)) for t in dom], label='OLD', color='blue')
                plt.plot(dom, [sqrt(self.nu(t, self.p, S1, W1, B1)) for t in dom], label='NEW', color='orange')
                if 0:
                    k = self.kernel
                    for b, s, w in zip(self.B, self.S, self.W):
                        Y = [b*k(x, y=w, p=self.p, s=s) for x in dom]
                        plt.plot(dom, Y, alpha=0.5, color='blue')


                plt.legend()
                plt.show(block=False)
                if 0:
                    if death: 
                        print(f'death index={j}')
                    elif update:
                        print(f'update index={j}')
                    else:
                        print(f'new birth b={b}')
                        print(f'new loc={w}')
                print(f'l0={l0}, l1={l1}')
                import pdb; pdb.set_trace()
            myplot()
        ######FOR DEBUG########


    @time_avg
    def sample_s(self):
        S1 = deepcopy(self.S)
        sold = S1[self.update_comp]
        snew = sold + norm.rvs()*self.s_proposal
        S1[self.update_comp] = snew

        l0 = self.l()
        l1 = self.l(S=S1)
        l = l1-l0
        prior_n = gamma.logpdf(snew, self.al, scale=1/self.bl)
        prior_d = gamma.logpdf(sold, self.al, scale=1/self.bl)
        prior = prior_n - prior_d
        prop = 0
        q = min([0, l+prior+prop])

        #print('sold={}, snew={}'.format(sold, snew))
        e = exponential(1)
        if e+q > 0: 
            self.S = deepcopy(S1)
            self.accepted['s'] += 1
            self.l0 = l1
        else:
            if self.l0 is None: self.l0 = l0

    @time_avg
    def sample_p(self):
        pold = self.p
        pnew = pold + norm.rvs()*self.p_proposal
        #pnew = exp(log(pold) + norm.rvs()*self.p_proposal)

        l0 = self.l()
        l1 = self.l(p=pnew)
        l = l1-l0
        prior_n = gamma.logpdf(pnew, self.ap, scale=1/self.bp)
        prior_d = gamma.logpdf(pold, self.ap, scale=1/self.bp)
        prior = prior_n - prior_d
        prop = 0
        q = min([0, l+prior+prop])

        #print('pold={}, pnew={}'.format(pold, pnew))
        e = exponential(1)
        if e+q > 0: 
            self.p = pnew
            self.accepted['p'] += 1
            self.l0 = l1
        else:
            if self.l0 is None: self.l0 = l0

    @time_avg
    def sample_w(self):
        W1 = deepcopy(self.W)
        wold = W1[self.update_comp]
        wnew = wold + norm.rvs()*self.w_proposal
        #if not 0 < wnew < 1: return

        W1[self.update_comp] = wnew

        l0 = self.l()
        l1 = self.l(W=W1)
        l = l1-l0
        q = min([0, l])

        #print('wold={}, wnew={}'.format(wold, wnew))
        e = exponential(1)
        if e+q > 0: 
            self.W = W1
            self.accepted['w'] += 1
            self.l0 = l1
        else:
            if self.l0 is None: self.l0 = l0

    def save(self, save):
        out = {
            'T': list(self.T),
            'X': list(self.X),
            'post': list(self.res)
            }
        with open(os.path.join(save, 'res.json'), 'w') as f:
            json.dump(out, f)

    @time_avg
    @timer
    def __call__(self, N=100, bip=0):
        assert N > bip
        self.accepted = {k: 0 for k in ['s','p', 'w', 'K']}
        N_update = 1e-8

        res = []

        self.S, self.J, self.W, self.B = {}, {}, {}, {}

        self.init()

        for i in range(N):
            self.iter = i
            progress(i, N, 'LARK', elapsed=time.time()-START)

            self.rj_mcmc()
            if self.update:
                N_update += 1
                self.sample_s()
                self.sample_w()
            self.sample_p()

            if i >= bip: res.append([self.p, self.S, self.J, self.W, self.B])

        self.res = res
        accept_pct = self.accepted['s']/N_update*100
        print(f'\nAcceptence [s] = {int(accept_pct)}%')
        accept_pct = self.accepted['w']/N_update*100
        print(f'\nAcceptence [w] = {int(accept_pct)}%')
        accept_pct = self.accepted['p']/N*100
        print(f'\nAcceptence [p] = {int(accept_pct)}%')
        accept_pct = self.accepted['K']/N*100
        print(f'\nAcceptence [K] = {int(accept_pct)}%')
        return res

@timer
def plot_out(posterior, lark, mtype='real', save=None, Treal=None, bip=0):
    m = len(lark.T)
    nu = lark.nu
    N = len(posterior)
    ps, ss = [], []

    dom = lark.T

    plot_post = []
    SUBS = False
    if SUBS:
        i4 = {1: 0, 100: 1, 2000: 2, 4000: 3}
        #i4 = {1000: 0, 4000: 1, 10000: 2, 40000: 3}
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('MCMC samples at different iterations')
        fig.tight_layout()
    start_plot = time.time()
    for i, post in enumerate(posterior):
        progress(i, N, 'Plotting', elapsed=time.time()-start_plot)
        if i < bip: continue
        p, S, J, W, B = post
        plot_post.append([sqrt(nu(x, p, S, W, B)*lark.dt) for x in dom])

        if SUBS:
            if i in i4 and mtype != 'real':
                j = i4[i]
                a, b = j//2, j%2
                ax[a, b].set_title(f'MCMC sample #{i}')
                ax[a, b].plot(dom, [getattr(Data, mtype)(x) for x in dom], label='True volatility', color='C1')
                ax[a, b].plot(dom, [sqrt(lark.nu(t, p, S, W, B)) for t in dom], label='MCMC sample', color='C0')
                ax[a, b].plot((W,W),([0 for w in W], [b for b in B]), c='black')

                if j == 3:
                    if save: savefig(save, 'MCMC_iters.pdf')
                    plt.figure() # make neater

    plot_post = matrix(plot_post)

    quantiles = []
    for i in range(m): quantiles.append(quantile(plot_post[:,i], [0.025, 0.975]))
    quantiles = matrix(quantiles)

    plot_post = array(plot_post.mean(0))[0]

    #RMSE################
    if mtype!='real':
        A = array([getattr(Data, mtype)(x) for x in lark.T])
        B = plot_post
        print('LARK RMSE = {}'.format(RMSE(A, B)))
    #RMSE################
    if Treal is not None:
        Tdom = Treal#[1:]
        plt.xticks(rotation=45)
    else:
        Tdom = dom

    plt.title('LARK')
    if mtype!='real':  plt.plot(Tdom, [getattr(Data, mtype)(x) for x in Tdom], label='True volatility', color='orange')

    plt.plot(Tdom, plot_post, label='Posterior Mean', color='C0')
    plt.fill_between(Tdom, array(quantiles[:, 0].flatten())[0], array(quantiles[:, 1].flatten())[0], alpha=0.5,
                     color='C0')

    plt.plot(Tdom, lark.X, alpha=0.5, label='Observations', color='black')
    plt.legend()
    if save: savefig(save, 'LARK.pdf')
    plt.figure() # make neater

    ##################
    Js = [J for _, _, J, _, _ in posterior]
    print('J mean = {}'.format(mean(Js)))
    plt.plot(Js, label='J trace', linewidth=0.25)
    if save: savefig(save, 'Jtrace.pdf')
    ##################

    plt.figure()
    ddd = linspace(0, 3, 1000)
    plt.title('p')
    ps = [p for p, _, _, _, _ in posterior]
    print('p mean = {}'.format(mean(ps)))
    plt.hist(ps, label=f'posterior', density=True, alpha=0.7, bins=30)
    plt.plot(ddd, gamma.pdf(ddd, lark.ap, scale=1/lark.bp), label='prior')
    plt.legend()
    if save: savefig(save, 'phist.pdf')

def main():
    global nomulti, cores, data, profile
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    parser.add_argument('--n', help='Sample size [100]', type=int)
    parser.add_argument('--N', help='MCMC iterations [1000]', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period [0]', type=int, default=0)
    parser.add_argument('--eps', help='epsilon [0.5]', type=float, default=0.5)
    parser.add_argument('--p', type=str, help='pb,pd,pu', default='0.4,0.4,0.2')
    parser.add_argument('--drift', type=str, default='zero')
    parser.add_argument('--ticker', type=str, help='ticker to get data', default='AAPL')
    parser.add_argument('--noplot', help='Plot output', action='store_true')
    parser.add_argument('--nomulti', help='no multiprocessing', action='store_true')
    parser.add_argument('--profile', help='profile methods', action='store_true')
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=os.cpu_count())
    parser.add_argument('--save', type=str, help='folder name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='kernel function', default='expon')
    parser.add_argument('--gentype', type=str, help='vol fn to use', default='sigt')
    parser.add_argument('--config', type=str, help='config with params based on gentype')
    args = parser.parse_args()

    profile = args.profile

    if args.config:
        with open(args.config) as f:
            params = yaml.safe_load(f)[args.gentype]
    else:
        params = {}

    kernel = params.get('kernel', args.kernel)
    eps = params.get('eps', args.eps)
    nu = params.get('nu', 1)
    vplus = params.get('vplus', 10)
    gammap = params.get('gammap', [1, 1])
    gammal = params.get('gammal', [1, 1])
    prop_bwsp = params.get('prop_bwsp', [1, 1, 1, 1])
    stable = params.get('stable', False)
    alpha = params.get('alpha')

    save = None
    if args.save:
        save = os.path.join(data, args.save)
        try:
            os.mkdir(save)
            os.mkdir(os.path.join(save, 'plots'))
        except:
            print('WARNING: {} probably already exists!!'.format(save))

    if args.load:
        load = os.path.join(data, args.load)
    nomulti = args.nomulti
    cores = args.cores

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    Treal = None
    if args.gentype=='real':
        T, X, Treal = Data.get_stock(n=args.n, ticker=args.ticker)
    else:
        T, X, dB= Data.gen_data_t(n=args.n, mtype=args.gentype)

    #####TEMP####
    try:
        with open(os.path.join(load, 'TX.json')) as fn:
            data = json.load(fn)
        res = data['post']
        T = data['T']
        X = data['X']
    except:
        print('No TX.json found')
    ############
    lark = LARK(T=T, X=X, p=p, eps=eps, kernel=kernel, drift=args.drift, 
                nu=nu, vplus=vplus, gammap=gammap, gammal=gammal, 
                proposals=prop_bwsp, stable=stable, alpha=alpha, nomulti=nomulti)
    if not args.load:
        res = lark(N=args.N)
    else:
        with open(os.path.join(load, 'res.json')) as fn:
            data = json.load(fn)
        res = data['post']
        lark.T = data['T']
        lark.X = data['X']
        lark.res = res

    if args.save: lark.save(save)
    if not args.noplot: plot_out(res, lark, mtype=args.gentype, save=save, Treal=Treal, bip=args.bip)
    plt.show()

if __name__=='__main__':
    main()
