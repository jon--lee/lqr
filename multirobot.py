from multilqr import InfMultiLQR
from highdim_lqr import HDLQR
import numpy as np
from vis import Visualizer
from utils import matmul, inBoundary 
"""
Robot
handles cost
handles lq regulation given dynamics
handles control reception
handles policy
handles target state and target actions
handles trajectory following
"""


class MultiRobot():
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

        self.lqr = InfMultiLQR(sys, self)
        

    def Qt(self, t = None):
        return self.Q
    def Rt(self, t = None):
        return self.R


    def reg_lti(self):
        self.lqr.converge1()
        self.lqr.converge2()
        return None 

    def control(self, u):
        x_p = self.sys.step(self.x - self.x_f, u - self.u_f)
        cost = self.cost(self.x - self.x_f, u - self.u_f)
        self.x = x_p + self.x_f
        return self.x, cost
    
    def pi(self, x, t):
        if inBoundary(x):
            K = self.lqr.K1t(t)
        else:
            K = self.lqr.K2t(t)
        return self.u_f + matmul(K, x - self.x_f)

    def cost(self, x, u):
        return (matmul(x.T, self.Q, x) + matmul(u.T, self.R, u))[0,0]
      
    def rollout(self, verbose=False):
        T = self.sys.T
        states = [self.x]
        controls = []
        costs = []
        if verbose:
            print "Initial state: " + str(self.x)
        for t in range(T):
            u = self.pi(self.x, t)
            controls.append(u)
            x, cost = self.control(u)
            if verbose:
                print ("\nt = " + str(t) + "\ncost: " + str(cost)
                        + "\ncontrol: " + str(u) + "\nstate: " + str(x))
            states.append(self.x)
            costs.append(cost)
        return states, controls, costs

    def rollout_learner(self, learner, verbose=False):
        T = self.sys.T
        states = [self.x]
        controls = []
        costs = []
        for t in range(T):
            u = learner.predict(self.x)
            controls.append(u)
            x, cost = self.control(u)
            if verbose:
                print ("\nt = " + str(t) + "\ncost: " + str(cost)
                        + "\ncontrol: " + str(u) + "\nstate: " + str(x))
            states.append(self.x)
            costs.append(cost)
        return states, controls, costs            


class HighDimRobot(MultiRobot):

    def __init__(self, sys, initial_state, T, Q = None, R = None, x_f = None, u_f = None):
        MultiRobot.__init__(self, sys, initial_state, T, Q, R, x_f, u_f)
        self.lqr = HDLQR(sys, self)


