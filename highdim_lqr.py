from multilqr import InfMultiLQR
from utils import matmul
import numpy as np

class HDLQR(InfMultiLQR):

    def converge1(self):
        P = self.robot.Qt()
        R = self.robot.Rt()
        Q = self.robot.Qt()
        A = self.sys.A1t()
        B = self.sys.B1t()
        t = 0
        while True and t < self.sys.T * 70:
            new_P = Q + matmul(A.T, P, A) - matmul(
                A.T, P, B, np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
            diff = new_P - P
            delta = np.max(np.abs(new_P - P))
            if delta < 1e-20:
                break
            P = new_P
            t += 1

        self.P1 = P
        self.K1 = -matmul(np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
        return

    def converge2(self):
        P = self.robot.Qt()
        R = self.robot.Rt()
        Q = self.robot.Qt()
        A = self.sys.A2t()
        B = self.sys.B2t()
        t = 0
        while True and t < self.sys.T * 70:
            new_P = Q + matmul(A.T, P, A) - matmul(
                A.T, P, B, np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
            diff = new_P - P
            delta = np.max(np.abs(new_P - P))
            if delta < 1e-20:
                break
            P = new_P
            t += 1

        self.P2 = P
        self.K2 = -matmul(np.linalg.inv(R + matmul(B.T, P, B)), B.T, P, A)
        return
