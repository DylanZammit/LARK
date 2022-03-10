from numpy import *

class Kernels:

    def GP_expon(self, x, y, gamma=1):
        return exp(-gamma*linalg.norm(x-y))

    def expon(self, x, y, p=1, s=0.05, **kwargs):
        return exp(-abs(x-y)**p/2/s)

    def aexpon(self, x, y, p=1, s=0.05, side='right', **kwargs):
        if side=='right' and x < y: return 0
        if side=='left' and x > y: return 0
        if x < y: return 0
        return exp(-abs(x-y)**p/s)

    def haar(self, x, y, s=0.1, **kwargs):
        return (abs(x-y)<=s)*1

    def haar_asym(self, x, y, s=0.1, **kwargs):
        return (0 < x-y <=s)*1

if __name__=='__main__':
    import matplotlib.pyplot as plt
    def plot_k(K, **kwargs):
        dom = linspace(0, 1, 1000)
        Y = [K(x=x, **kwargs) for x in dom]
        plt.plot(dom, Y)
        plt.show(block=False)

    kk = Kernels()
    K = kk.aexpon
    plot_k(K, y=0, p=1, s=0.05)
    import pdb; pdb.set_trace()
