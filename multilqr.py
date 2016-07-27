import numpy as np
from utils import matmul




"""
Infinite horizon LQR assuming that dynamics
and cost are linear time invariant
"""


class InfMultiLQR():

    def __init__(self, sys, robot):
        self.K1 = None
        self.P1 = None
        self.K2 = None
        self.P2 = None
        self.sys = sys
        self.robot = robot

    def K1t(self, t=None):
        return self.K1

    def K2t(self, t=None):
        return self.K2

    def P1t(self, t=None):
        return self.P1
    def P2t(self, t=None):
        return self.P2

    def loadKs(self):
        self.K1 = np.load("data/k1.npy")
        self.K2 = np.load("data/k2.npy")

    def saveKs(self):
        np.save("data/k1.npy", self.K1)
        np.save("data/k2.npy", self.K2)

    def converge1(self):
        P = self.robot.Qt()
        R = self.robot.Rt()
        Q = self.robot.Qt()
        A = self.sys.A1t()
        B = self.sys.B1t()
        while True:
            new_P = Q + matmul(A.T, P, A) - matmul(
                A.T, P, B, np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
            diff = new_P - P
            delta = np.max(np.abs(new_P - P))
            if delta < 1e-2:
                break
            P = new_P

        self.P1 = P
        self.K1 = -matmul(np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
        return

    def converge2(self):
        P = self.robot.Qt()
        R = self.robot.Rt()
        Q = self.robot.Qt()
        A = self.sys.A2t()
        B = self.sys.B2t()
        while True:
            new_P = Q + matmul(A.T, P, A) - matmul(
                A.T, P, B, np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
            diff = new_P - P
            delta = np.max(np.abs(new_P - P))
            if delta < 1e-2:
                break
            P = new_P

        self.P2 = P
        self.K2 = -matmul(np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
        return
    
