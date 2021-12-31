from numpy import *

class Kernels:

    def expon(self, x, y, p=1, s=0.05):
        return exp(-abs(x-y)**p/s)

    def expon_asym(self, x, y, p=1, s=0.05):
        if x > y: return 0
        return exp(-abs(x-y)**p/s)

    def haar(self, x, y, s=0.1):
        return (abs(x-y)<=s)*1

    def haar_asym(self, x, y, s=0.1):
        return (0 < x-y <=s)*1 # asymmetric?

