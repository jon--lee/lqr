from lqr2 import LQR
from sys2 import SystemLTI
import numpy as np
from utils import matmul
"""
Robot
handles cost
handles lq regulation given dynamics
handles control reception
handles policy
handles target state and target actions
handles trajectory following
"""


class RobotLTI():
    def __init__(self, sys, initial_state, T, Q = None, R = None, x_f = None, u_f = None):
        self.INITIAL_STATE = self.x = initial_state
        self.sys = sys
        sys.add_robot(self)
        
        if Q is None:
            Q = np.zeros((sys.xdims, sys.xdims))
        self.Q = Q
        if R is None:
            R = np.zeros((sys.udims, sys.udims))
        self.R = R

        if x_f is None:
            x_f = np.zeros((sys.xdims, 1))
        self.x_f = x_f
        if u_f is None:
            u_f = np.zeros((sys.udims, 1))
        self.u_f = u_f

        self.lqr = LQR(sys, self, sys.T)
        

    def Qt(self, t = None):
        return self.Q
    def Rt(self, t = None):
        return self.R


    def reg_lti(self):
        self.lqr.converge()
        return self.lqr.K[0]

    def control(self, u):
        x_p = self.sys.step(self.x, u)
        cost = self.cost(self.x, u)
        self.x = x_p
        return x_p, cost
    
    def pi(self, x, t):
        return matmul(self.lqr.K[t], x)

    def cost(self, x, u):
        return (matmul(x.T, Q, x) + matmul(u.T, R, u))[0,0]
       

if __name__ == '__main__':
    T = 1000
    xdims = 2
    udims = 1
    
    A = np.array([[ 1, 2], [3, 4]])
    B = np.array([[3], [4]])

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])

    init_state = np.array([[1], [2]])

    sys = SystemLTI(xdims, udims, T, A, B)
    robot = RobotLTI(sys, init_state, T, Q, R)
    robot.reg_lti()

    print robot.x
    for i in range(T):
        u = robot.pi(robot.x, i)
        cost, x = robot.control(u)
        print cost, x
    
