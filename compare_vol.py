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
from common import *
from getdata import Data
from LARK_vol import LARK, plot_out
from GP_vol import GP, plot_gp
from gugu_vol import Gugu, plot_gugu
from kernels import Kernels

data = '/home/dylan/git/LARK/data'

def main():
    #global nomulti, cores, data
    global data
    parser = argparse.ArgumentParser(description='MCMC setup args.')
    #LARK
    parser.add_argument('--n', help='Sample size [100]', type=int, default=100)
    parser.add_argument('--N', help='MCMC iterations [1000]', type=int, default=1000)
    parser.add_argument('--bip', help='MCMC burn-in period [0]', type=int, default=0)
    parser.add_argument('--eps', help='epsilon [0.5]', type=float, default=0.5)
    parser.add_argument('--p', type=str, help='pb,pd,pu', default='0.4,0.4,0.2')
    parser.add_argument('--drift', type=str, default='zero')
    parser.add_argument('--ticker', type=str, help='ticker to get data', default='AAPL')
    parser.add_argument('--noplot', help='Plot output', action='store_true')
    parser.add_argument('--nomulti', help='no multiprocessing', action='store_true')
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=os.cpu_count())
    parser.add_argument('--save', type=str, help='folder name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='kernel function', default='expon')
    parser.add_argument('--gentype', type=str, help='vol fn to use', default='sigt')
    parser.add_argument('--gugum', help='Gugu bin width', type=int, default=100)
    parser.add_argument('--gpgam', help='GP gamma param', type=int, default=11)
    parser.add_argument('--config', type=str, help='config with params based on gentype')
    parser.add_argument('--nolark', help='Plot output', action='store_true')
    parser.add_argument('--nogp', help='Plot output', action='store_true')
    parser.add_argument('--nogugu', help='Plot output', action='store_true')
    args = parser.parse_args()

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

    nomulti = args.nomulti
    cores = args.cores

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

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    Treal = None
    if args.gentype=='real':
        T, X, Treal = Data.get_stock(n=args.n, ticker=args.ticker)
    else:
        T, X, dB = Data.gen_data_t(n=args.n, mtype=args.gentype)

    if args.load:
        with open(os.path.join(load, 'res.json')) as fn:
            data = json.load(fn)
        T = array(data['T'])
        X = array(data['X'])
        res = data['post']

    lark = LARK(T=T, X=X, p=p, eps=eps, kernel=kernel, drift=args.drift, 
                nu=nu, vplus=vplus, gammap=gammap, gammal=gammal, proposals=prop_bwsp,
               nomulti=nomulti, cores=cores, stable=stable, alpha=alpha)

    if not args.nolark:
        if not args.load:
            print('Running LARK method...', end='')
            res = lark(N=args.N)
            print('done')
        else:
            print('Loading LARK method...', end='')
            lark.res = res
            print('done')
        if args.save: lark.save(save)
        if not args.noplot: 
            plot_out(res, lark, mtype=args.gentype, save=save, Treal=Treal, bip=args.bip)
        plt.figure()

    T, X = lark.T, lark.X
    if not args.nogp:
        print('\nRunning GP method...', end='')
        if isinstance(X, list): X = array(X)
        myK = Kernels().GP_expon
        K = lambda x, y: myK(x, y, gamma=args.gpgam)
        #K = myK
        alpha = -1.27036 # these should depend on dt
        beta = sqrt(pi**2/2)
        Z = log(X**2)/beta

        gp = GP(T, Z, K, sig=1) # change sig
        if not args.noplot: 
            plot_gp(gp, X, args.gentype, Treal=Treal)
            if args.save: savefig(args.save, 'GP.pdf')
        plt.figure()
        print('done')

    if not args.nogugu:
        print('Running GUGU method...', end='')
        model = Gugu(X, m=args.gugum)
        if not args.noplot:
            plot_gugu(model, T, args.gentype, Treal=Treal)
            if args.save: savefig(args.save, 'gugu.pdf')
        print('done')

    plt.show()


if __name__ == '__main__':
    main()
