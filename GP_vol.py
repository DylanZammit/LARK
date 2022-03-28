import numpy as np
import matplotlib.pyplot as plt
from getdata import Data
from common import RMSE

def K(x, y, gamma=1):
    return np.exp(-gamma*np.linalg.norm(x-y))

def plot_gp(gp, Z, gentype='sigt', Treal=None):
    T, X = gp.X, gp.Y
    n = len(T)
    alpha = -1.27036# -np.log(len(T))# these should depend on dt
    beta = np.pi**2/2
    g = lambda x: np.sqrt(np.exp(x*beta-alpha)*len(T))
    mean = g(gp.E(T))
    lower = g(np.diag(gp.band(T, False)))
    upper = g(np.diag(gp.band(T)))

    mean, lower, upper = mean/np.sqrt(n), lower/np.sqrt(n), upper/np.sqrt(n)
    if Treal is not None:
        Tdom = Treal
        plt.xticks(rotation=45)
    else:
        Tdom = T
    print('done')
    plt.title('GP Method')
    plt.plot(Tdom, Z, color='black', alpha=0.4)
    if gentype != 'real': 
        plt.plot(T, [getattr(Data, gentype)(t) for t in T], label='True Volatility', color='orange')
    plt.plot(Tdom, mean, label='GP posterior', color='blue')
    plt.fill_between(Tdom, lower, upper, alpha=0.2, color='blue')
    plt.legend()
    #RMSE################
    if gentype!='real':
        A = np.array([getattr(Data, gentype)(x) for x in T])
        B = g(gp.E(T))/np.sqrt(n)
        print('\nGP RMSE = {}'.format(RMSE(A, B)))
    #RMSE################

class Model:

    def f(self, x): 
        return np.sqrt(x)

    def s(self, x, c=3/2):
        return c + np.sin(2*(4*x-2)) + 2*np.exp(-16*(4*x-2)**2)

    def gen_data(self, X):
        Y = np.array([self.f(x) + np.random.randn()*self.s(x) for x in X])
        return Y

class GP:
    def __init__(self, X, Y, K, sig=1):
        assert len(X)==len(Y)
        n = len(X)
        self.n = n
        G = np.matrix([[K(xi,xj) for xi in X] for xj in X])

        b = np.linalg.inv(G+sig**2*np.eye(n))
        abar = b@Y
        abar = np.array(abar)[0]

        self.X = X
        self.Y = Y

        self.sig = sig
        self.b = b
        self.G = G
        self.K = K
        self.abar = abar

    def Gram(self, D):
        return np.array([[self.K(xi, xj) for xi in D] for xj in D])

    def GramData(self, D):
        return np.array([[self.K(xi, xj) for xi in D] for xj in self.X])

    def V(self, D):
        G = self.Gram(D)
        Gstar = self.GramData(D)
        
        return G-Gstar@self.b@Gstar

    def E(self, D):
        if isinstance(D, float): D = [D]

        k = self.GramData(D)
        return k@self.abar

    def band(self, D, upper=True):
        sg = 1 if upper else -1
        return 1.96*sg*np.sqrt(self.V(D))+self.E(D)

if __name__ == '__main__':
    n = 400

    model = Model()
    X = np.linspace(0, 1, n)
    Y = model.gen_data(X)
    Y2 = pow(Y, 2)

    drift_gp = GP(X, Y, K)
    mu = drift_gp.E(X)

    alpha = -1.27036284-np.log(n)
    beta = np.pi**2/2
    Z = np.log((Y-mu)**2)/beta
    def g(x): return np.exp(x*beta-alpha)

    log_gp = GP(X, Z, K)

    dom = X*n

    plt.plot(dom, model.s(X)**2, label='$\sigma_t^2$', color='blue')

    all_gp = GP(X, Y2, K)

    mean = all_gp.E(X)
    plt.plot(dom, mean, label='Batz', color='orange')

    mean = g(log_gp.E(X))
    lower = g(np.diag(log_gp.band(X, False)))
    upper = g(np.diag(log_gp.band(X)))
    plt.plot(dom, mean, label='LogExp', color='red')
    plt.fill_between(dom, lower, upper, alpha=0.2, color='red')

    if 0: plt.scatter(dom, Y2, label = '$Y^2$')

    plt.legend()
    plt.show()
