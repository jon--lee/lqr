from sys2 import SystemLTI, SystemPointMass
from robot import RobotLTI
from env import point_mass as pm
from vis import Visualizer
import numpy as np

if __name__ == '__main__':
    T = 30
    xdims = pm.xdims
    udims = pm.udims
    
    A = pm.A
    B = pm.B
    Q = pm.Q
    R = pm.R
    init_state = pm.init_state
    
    x_f = np.array([[10], [2], [0], [0]])
    u_f =  np.array([[0], [0]])

    sys = SystemPointMass(xdims, udims, T, A, B, stoch=False)
    robot = RobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    
    const_u = np.array([[10], [10]])
    
    x = robot.x
    art_states = [robot.x]
    art_controls = []
    for t in range(30):
        u = np.array([[0], [1]])
        x = sys.dyn(x, u)
        art_controls.append(u)
        art_states.append(x)


    T = 70
    xdims = pm.xdims
    udims = pm.udims
    
    A = pm.A
    B = pm.B
    Q = pm.Q
    R = pm.R
    init_state = art_states[-1]
    
    x_f = np.array([[10], [2], [0], [0]])
    u_f =  np.array([[0], [0]])

    sys = SystemPointMass(xdims, udims, T, A, B, stoch=False)
    robot = RobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    states, controls = robot.rollout(verbose=False)
    art_states = art_states[:-1] + states
    art_controls = art_controls + controls

    








