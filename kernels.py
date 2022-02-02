from numpy import *

class Kernels:

    def expon(self, x, y, p=1, s=0.05, **kwargs):
        return exp(-abs(x-y)**p/s)

    def aexpon(self, x, y, p=1, s=0.05, side='right', **kwargs):
        if side=='right' and x < y: return 0
        if side=='left' and x > y: return 0
        if x < y: return 0
        return exp(-abs(x-y)**p/s)

    def haar(self, x, y, s=0.1, **kwargs):
        return (abs(x-y)<=s)*1

    def haar_asym(self, x, y, s=0.1, **kwargs):
        return (0 < x-y <=s)*1

