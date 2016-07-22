from robot import RobotLTI
from sys2 import SystemLTI
from env import basic2x1
from vis import Visualizer
import numpy as np

if __name__ == '__main__':
    T = 100
    xdims = 2
    udims = 1

    A = basic2x1.A
    B = basic2x1.B
    Q = basic2x1.Q
    R = basic2x1.R
    init_state = basic2x1.init_state
    
    x_f = np.array([[0], [0]])
    u_f =  np.array([[0]])

    sys = SystemLTI(xdims, udims, T, A, B, stoch=True)
    robot = RobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    states = robot.rollout(verbose=True)
    
    vis = Visualizer()
    vis.set_recording(states)
    vis.set_target(x_f)
    vis.show()
