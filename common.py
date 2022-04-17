from time import time
from scipy.stats import poisson, gamma, norm, nbinom, chi2
from scipy.special import exp1
from scipy.stats import rv_continuous, pareto
from numpy import *
import matplotlib.pyplot as plt
import os

data = '/home/dylan/git/LARK/data'
def expinv(x, d=1e-9):
    return chi2.ppf(1-x*d/2, d)/2

def plot_pareto(size=1000):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pareto

    x_m, b = 0.1, 0.5
    X = pareto.rvs(b, size=size, scale=x_m)
    X = X[X<6]
    dom = np.linspace(0, 6, 10000)
    plt.hist(X, alpha=0.5, bins=100, density=True)
    plt.plot(dom, [pareto.pdf(x, scale=x_m, b=b) for x in dom])
    plt.show()

def rvs_test(self, size=1000):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import trapz

    X = self.rvs(size=size)
    dom = np.linspace(0, 6, 10000)
    print('Integral form eps to inf = {}'.format(trapz([self.pdf(x) for x in dom], dom)))
    plt.hist(X, alpha=0.5, bins=100, density=True)
    plt.plot(dom, [self.pdf(x) for x in dom])
    plt.show()

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
            raise Exception
        return log(self.alpha)-self.beta*x

    def _ppf(self, x):
        return expinv(exp1(self.eps)*(1-x))


class Pareto:

    def __init__(self, eps, nu, alpha):
        self.eps = eps/nu
        self.alpha = alpha

    def logpdf(self, x):
        return pareto.logpdf(x, scale=self.eps, b=self.alpha)

    def rvs(self, size=1):
        X = pareto.rvs(self.alpha, scale=self.eps, size=size)
        return X[0] if len(X) == 1 else X

class SaS(rv_continuous):

    def __init__(self, eps, alpha, nu, *args, **kwargs):
        self.eps = eps/nu
        self.alpha = alpha
        #self.c = math.gamma(alpha)*sin(alpha*pi/2)/pi
        self.c = alpha*pow(self.eps, alpha)
        self.a = self.eps
        assert alpha > 0
        assert nu > 0
        assert eps > 0
        super().__init__(*args, **kwargs)

    def _pdf(self, x):
        return self.c/x**(1+self.alpha)

    def _logpdf(self, x):
        return log(self.c)-(1+self.alpha)*log(x)

    def _cdf(self, x):
        return 1-(self.eps/x)**self.alpha

    def _ppf(self, x):
        return 0 if x < self.eps else self.eps*pow(1-x, -1/self.alpha)
        #return pow(x/self.eps**self.alpha+self.eps**self.alpha, 1/self.alpha)

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

def progress(i, N, title='', elapsed=None):
    if int(i/N*100)!=int((i-1)/N*100):
        if elapsed: 
            unit='s'
            ETA = (elapsed*N/i-elapsed)
            if ETA > 60: 
                ETA/=60
                unit = 'm'
            if ETA > 60: 
                ETA/=60
                unit = 'h'

            print(f'{title}: {int(i/N*100)}%...ETA={ETA:.2f}{unit}', end='\r')
        else:
            print(f'{title}: {int(i/N*100)}%', end='\r')

def savefig(save, name):
    fn = os.path.join(data, save, 'plots', name)
    plt.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=1000, format='pdf')

def RMSE(x, y):
    return sqrt(sum((x-y)**2)/len(x))
