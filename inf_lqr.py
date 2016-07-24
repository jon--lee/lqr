"""
Infinite horizon LQR assuming that dynamics
and cost are linear time invariant
"""


from lqr import LQR
from utils import matmul
import numpy as np
class InfLQR(LQR):

    def __init__(self, sys, robot):
        self.K = None
        self.P = None
        self.sys = sys
        self.robot = robot

    def Kt(self, t=None):
        return self.K

    def Pt(self, t=None):
        return self.P


    def converge(self):
        P = self.robot.Qt()
        R = self.robot.Rt()
        Q = self.robot.Qt()
        A = self.sys.At()
        B = self.sys.Bt()
        while True:
            new_P = Q + matmul(A.T, P, A) - matmul(
                A.T, P, B, np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
            diff = new_P - P
            delta = np.max(np.abs(new_P - P))
            if delta < 1e-2:
                break
            P = new_P

        self.P = P
        self.K = -matmul(np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
        return