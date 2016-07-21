import numpy as np
from lqr import LQR
class Robot():

    def __init__(self, sys, xdims, udims, T):
        self.xdims = xdims
        self.udims = udims
        self.sys = sys
        self.x = np.zeros((self.xdims, 1))
        self.lqr = LQR(xdims, udims, T)
        self.T = T
        return

    def reg_lti(self, A, B, Q, R):
        self.lqr.converge(A, B, Q, R)
        print self.lqr.K[0]

    def control(self, u):
        x_p, cost = self.sys.step(self.x, u)
        self.x = x_p
        return x_p, cost

    def pi(self, x, t):
        return np.matmul(self.lqr.K[t], x)
        
if __name__ == '__main__':
    A = np.array([ 1, 2], [3, 4])
    B = np.array([[3, 4]])
    print "A: " + str(A)
    print "B: " + str(B)
    
