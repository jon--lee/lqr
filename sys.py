from utils import matmul
import numpy as np  
"""
handles dynamics
handles steps
handles noise
handles time
"""


class SystemLTI():

    def __init__(self, xdims, udims, T, A = None, B = None, w = None):
        self.xdims = xdims
        self.udims = udims
        
        self.T = T
        self.t = 0

        self.robot = None

        # Setting dynamics
        if A is None:
            A = np.zeros((self.xdims, self.xdims))
        self.A = A
        if B is None:
            B = np.zeros((self.xdims, self.udims))
        self.B = B

        self.mean = np.zeros(self.xdims)
        self.cov = np.identity(self.xdims) * 2

    def At(self, t = None):
        return self.A
    def Bt(self, t = None):
        return self.B
    def wt(self):
        w = np.random.multivariate_normal(self.mean, self.cov, 1)
        w = np.reshape(w, (self.xdims, 1))
        print w
        return w
        #return np.zeros((self.xdims, 1))

    def reset_robot(self):
        self.robot.x = self.robot.INITIAL_STATE
        self.t = 0

    def add_robot(self, robot):
        self.robot = robot
        self.reset_robot()

    def dyn(self, x, u):
        A = self.At()
        B = self.Bt()
        w = self.wt()
        return matmul(A, x) + matmul(B, u) + w

    def step(self, x, u):
        x_p = self.dyn(x, u)
        self.t += 1
        return x_p
        

