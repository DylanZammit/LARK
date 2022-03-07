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
from common import *
from getdata import Data
from LARK_vol import LARK, plot_out
from GP_vol import GP, plot_gp
from gugu_vol import Gugu, plot_gugu
from kernels import Kernels


def main():
    #global nomulti, cores, data
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
    parser.add_argument('--plot_samples', help='Plot output', action='store_true')
    parser.add_argument('--save', type=str, help='folder name to save to', default=None)
    parser.add_argument('--load', type=str, help='file name to load from', default=None)
    parser.add_argument('--kernel', type=str, help='comma separated kernel functions', default='expon')
    parser.add_argument('--gentype', type=str, help='vol fn to use', default='sigt')
    args = parser.parse_args()

    #raises exception if exists
    save = None
    if args.save:
        save = os.path.join(data, args.save)
        os.mkdir(save)
    elif args.load:
        load = os.path.join(data, args.load)

    nomulti = args.nomulti
    cores = args.cores

    p = tuple([float(x) for x in args.p.split(',')])
    assert isclose(sum(p), 1)

    Treal = None
    if not args.load:
        if args.gentype=='real':
            T, X, dB = Data.get_stock(n=args.n, ticker=args.ticker)
        else:
            T, X, Treal = Data.gen_data_t(n=args.n, mtype=args.gentype)

        lark = LARK(T=T, X=X, p=p, eps=args.eps, kernel=args.kernel.split(','), drift=args.drift, nomulti=nomulti, cores=cores)
        print('Running LARK method...', end='')
        res = lark(N=args.N, bip=args.bip)
        print('done')
    else:
        print('Loading LARK method...', end='')
        with open(os.path.join(load, 'res.json')) as fn:
            data = json.load(fn)
        res = data['post']
        lark.T = data['T']
        lark.X = data['X']
        lark.res = res
        print('done')
    if not args.noplot: plot_out(res, lark, args.plot_samples, mtype=args.gentype, save=save)
    plt.figure()

    print('Running GP method...', end='')
    T, X = lark.T, lark.X
    K = Kernels().GP_expon
    alpha = -1.27036 # these should depend on dt
    beta = pi**2/2
    #g = lambda x: exp(x*beta-alpha)
    Z = log(X**2)/beta

    gp = GP(T, Z, K, sig=1) # change sig
    if not args.noplot: plot_gp(gp, X, args.gentype)
    plt.figure()
    print('done')

    print('Running GUGU method...', end='')
    model = Gugu(X, m=5)
    plot_gugu(model, T, args.gentype)
    print('done')

    plt.show()


if __name__ == '__main__':
    main()
