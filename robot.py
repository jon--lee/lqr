from lqr import LQR
import numpy as np
from vis import Visualizer
from sys2 import SystemLTI
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
        x_p = self.sys.step(self.x - self.x_f, u - self.u_f)
        cost = self.cost(self.x - self.x_f, u - self.u_f)
        self.x = x_p + self.x_f
        return self.x, cost
    
    def pi(self, x, t):
        return self.u_f + matmul(self.lqr.K[t], x - self.x_f)

    def cost(self, x, u):
        return (matmul(x.T, self.Q, x) + matmul(u.T, self.R, u))[0,0]
      
    def rollout(self, verbose=False):
        T = self.sys.T
        states = [self.x]
        for t in range(T):
            u = self.pi(self.x, t)
            x, cost = self.control(u)
            if verbose:
                print ("\nt = " + str(t) + "\ncost: " + str(cost)
                        + "\ncontrol: " + str(u) + "\nstate: " + str(x))
            states.append(self.x)
        return states

if __name__ == '__main__':
    T = 100
    xdims = 2
    udims = 1
    
    A = np.array([[ 1, 2], [3, 4]])
    B = np.array([[3], [4]])

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])

    init_state = np.array([[-10], [15]])

    x_f = np.array([[0], [0]])
    u_f = np.array([[0]])

    sys = SystemLTI(xdims, udims, T, A, B, stoch=False)
    #init_state = sys.initial_state(init_state, noise=True)
    robot = RobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    print robot.x
    states = [robot.x]
    for t in range(T):
        u = robot.pi(robot.x, t)
        x, cost = robot.control(u)
        print "\n"
        print "t = " + str(t)
        print "cost: " + str(cost)
        print "control: " + str(u)
        print "state: " + str(x)
        states.append(robot.x)


    vis = Visualizer()
    vis.set_recording(states)
    vis.set_target(x_f)
    vis.show()
