import numpy as np
from utils import matmul

class LQR():

    def __init__(self, sys, robot, T):
        self.sys = sys
        self.robot = robot
        self.xdims = sys.xdims
        self.udims = sys.udims
        self.T = T
        self.K = [None]*T
        self.P = [None]*(T+1)
        
    def Kt(self, t):
        return self.K[t]

    def Pt(self, t):
        return self.P[t]

    def converge(self):
        P = self.P
        P[self.T] = self.robot.Qt(self.T)
        for t in range(self.T - 1, -1 , -1):
            R = self.robot.Rt(t)
            Q = self.robot.Qt(t)
            A = self.sys.At(t)
            B = self.sys.Bt(t)
            self.K[t] = -matmul(np.linalg.inv(R + matmul(B.T, P[t+1], B)), B.T, P[t+1], A)
            self.P[t] = Q + matmul(A.T, P[t+1], A) - matmul(
                A.T, P[t+1], B, np.linalg.inv(R + matmul(B.T, P[t+1], B)), B.T, P[t+1], A)
        return
