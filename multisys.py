from sys2 import SystemLTI
from utils import matmul, inBoundary
import numpy as np

class PiecewiseLinearSys():

    def __init__(self, xdims, udims, T, A1=None, B1=None, A2=None, B2=None, stoch=False):
        self.xdims = xdims
        self.udims = udims
        self.T = T
        self.t = 0

        self.robot = None

        if A1 is None:
            A1 = np.zeros((self.xdims, self.xdims))
        if A2 is None:
            A2 = np.zeros((self.xdims, self.xdims))
        if B1 is None:
            B1 = np.zeros((self.xdims, self.udims))
        if B2 is None:
            B2 = np.zeros((self.xdims, self.udims))
        self.A1 = A1
        self.A2 = A2
        self.B1 = B1
        self.B2 = B2
        
        self.stoch = stoch
        self.mean = np.zeros(self.xdims)
        self.cov_init = np.identity(self.xdims) * 20
        self.cov = np.identity(self.xdims) * .00001#.1

    def A1t(self, t=None):
        return self.A1
    def A2t(self, t=None):
        return self.A2
    def B1t(self, t=None):
        return self.B1
    def B2t(self, t=None):
        return self.B2

    def wt(self):
        if self.stoch:
            w = np.random.multivariate_normal(self.mean, self.cov, 1)
            w = np.reshape(w, (self.xdims, 1))
            return w
        else:
            return np.zeros((self.xdims, 1))

    def initial_state(self, x, noise = False):
        if not noise:
            return x
        w = np.random.multivariate_normal(self.mean, self.cov_init, 1)
        w = np.reshape(w, (self.xdims, 1))
        return x + w

    def reset_robot(self):
        self.robot.x = self.robot.INITIAL_STATE
        self.t = 0
    
    def add_robot(self, robot):
        self.robot = robot
        self.reset_robot()

    def dyn(self, x, u):
        if inBoundary(x):
            A = self.A1t()
            B = self.B1t()
        else:
            A = self.A2t()
            B = self.B2t()
        w = self.wt()
        return matmul(A, x) + matmul(B, u) + w

    def step(self, x, u):
        x_p = self.dyn(x, u)
        self.t += 1
        return x_p


    

    
