from sys2 import SystemLTI, SystemPointMass
from env import point_mass as pm
from vis import Visualizer
from robot import RobotLTI

import numpy as np

if __name__ == '__main__':
    T = 100
    xdims = pm.xdims
    udims = pm.udims
    
    A = pm.A
    B = pm.B
    Q = pm.Q
    R = pm.R
    init_state = pm.init_state
    
    x_f = np.array([[10], [2], [0], [0]])
    u_f =  np.array([[0], [0]])

    sys = SystemPointMass(xdims, udims, T, A, B, stoch=True)
    robot = RobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    states = robot.rollout(verbose=True)
    
    vis = Visualizer()
    vis.set_recording(states)
    vis.set_target(x_f)
    vis.show()

