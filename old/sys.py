"""
Time invariant system
Linear
"""

import numpy as np
from robot import Robot

class SystemLTI():

    def __init__(self, xdims, udims, T):
        self.t = 0
        self.xdims = xdims
        self.udims = udims
        self.A = np.zeros((self.xdims, self.xdims))
        self.B = np.zeros((self.xdims, self.udims))
        self.Q = np.zeros((self.xdims, self.xdims))
        self.R = np.zeros((self.udims, self.udims))
        self.w = np.zeros((self.xdims, 1))
        self.robot = Robot(self, self.xdims, self.udims, T)
        self.reset_robot()
        self.T = T
        self.x_f = np.zeros((xdims, 1))
        self.u_f = np.zeros((xdims, 1))

    def reset_robot(self):
        if not self.robot:
            self.robot = Robot(self, self.xdims, self.udims, self.T)
        self.robot.x = np.zeros((self.xdims, self.udims)) + 1
        self.t = 0

    def setA(self, A):
        self.A = A

    def setB(self, B):
        self.B = B

    def At(self):
        return self.A

    def Bt(self):
        return self.B

    def setQ(self, Q):
        self.Q = Q

    def setR(self, R):
        self.R = R

    def Qt(self):
        return self.Q

    def Rt(self):
        return self.R

    def dyn(self, x, u, A, B):
        x = np.array(x)
        u = np.array(u)
        A = np.array(A)
        B = np.array(B)
        return np.matmul(A, x) + np.matmul(B, u) + self.w

    def cost(self, x, u, Q, R):
        x = np.array(x)
        u = np.array(u)
        Q = np.array(Q)
        R = np.array(R)        
        q = np.matmul(np.matmul((self.x_f - x).T, Q), self.x_f - x)
        r = np.matmul(np.matmul((self.u_f - u).T, R), self.u_f - u)
        return (q + r)[0,0]

    def step(self, x, u):
        A = self.At()
        B = self.Bt()
        Q = self.Qt()
        R = self.Rt()
        self.t += 1
        return self.dyn(x, u, A, B), self.cost(x, u, Q, R)
    
if __name__ == '__main__':
    T = 1000
    
    A = np.array([[ 1, 2], [3, 4]])
    B = np.array([[3], [4]])
    print "A: " + str(A)
    print "B: " + str(B)
    print "\n"

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])
    print "Q: " + str(Q)
    print "R: " + str(R)
    print "\n"

    xdims = 2
    udims = 1

    sys = SystemLTI(xdims, udims, T)
    sys.setA(A)
    sys.setB(B)
    sys.setQ(Q)
    sys.setR(R)
    robot = sys.robot
    robot.reg_lti(sys.A, sys.B, sys.Q, sys.R)


    print robot.x
    for i in range(T):
        u = robot.pi(robot.x, i)
        cost, x = robot.control(u)
        print cost, x

