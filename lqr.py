import numpy as np

def mult(*args):
    product = args[0]
    for A in args[1:]:
        product = np.matmul(product, A)
    return product

class LQR():

    def __init__(self, xdims, udims, T):
        self.xdims = xdims
        self.udims = udims
        self.T = T
        self.K = [None]*T
        self.P = [None]*(T+1)
        

    def converge(self, A, B, Q, R):
        P = self.P
        P[self.T] = Q
        for t in range(self.T - 1, -1 , -1):
            #self.K[t] = mult(B.T)
            #self.K[t] = -mult(np.linalg.inv(R + mult(B.T, P[t+1], B)))
            self.K[t] = -mult(np.linalg.inv(R + mult(B.T, P[t+1], B)), B.T, P[t+1], A)
            self.P[t] = Q + mult(A.T, P[t+1], A) - mult(
                A.T, P[t+1], B, np.linalg.inv(R + mult(B.T, P[t+1], B)), B.T, P[t+1], A)
        return
