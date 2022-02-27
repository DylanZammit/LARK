from time import time
from scipy.stats import poisson, gamma, norm, nbinom, chi2
from scipy.special import exp1
from scipy.stats import rv_continuous
from numpy import *

def expinv(x, d=1e-9):
    return chi2.ppf(1-x*d/2, d)/2

class Gamma(rv_continuous):

    def __init__(self, eps, nu, *args, **kwargs):
        self.nu = nu
        self.eps = eps
        self.a = eps/nu
        assert nu > 0
        super().__init__(a=self.a, **kwargs)

    def _pdf(self, x):
        return exp(-x*self.nu)/x/exp1(self.eps) if x > self.a else 0

    def _logpdf(self, x):
        return -log(x)-x*self.nu-log(exp1(self.eps))

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


def timer(func):
    def wrapper_function(*args, **kwargs):
        t0 = time()
        print(f'\n Running {func.__name__}')
        out = func(*args,  **kwargs)
        seconds = int(time()-t0)
        minutes = round(seconds/60, 1)
        print(f'\n{func.__name__} time taken: {seconds}s={minutes}mins')
        return out
    return wrapper_function

def progress(i, N, title=''):
    if int(i/N*100)!=int((i-1)/N*100):
        print(f'{title}: {int(i/N*100)}%', end='\r')
