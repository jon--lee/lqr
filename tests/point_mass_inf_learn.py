from sys2 import SystemLTI, SystemPointMass
from env import point_mass as pm
from vis import Visualizer
from robot import RobotLTI, InfRobotLTI
from sklearn.linear_model import LinearRegression
import IPython
from sk_learner import SKLearner
import numpy as np






if __name__ == '__main__':
    T = 50
    xdims = pm.xdims
    udims = pm.udims
    
    A = pm.A
    B = pm.B
    Q = pm.Q
    R = pm.R
    init_state = pm.init_state
    
    x_f = np.array([[0], [0], [0], [0]])#x_f = np.array([[10], [2], [0], [0]])
    u_f =  np.array([[0], [0]])

    sys = SystemPointMass(xdims, udims, T, A, B, stoch=True)
    robot = InfRobotLTI(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    sup_states, sup_controls, sup_costs = robot.rollout(verbose=False)
    
    vis = Visualizer()
    vis.set_recording(sup_states)
    vis.set_target(x_f)
    vis.show()

    print "LQR cost: " + str(sum(sup_costs))

    """lr = LinearRegression(fit_intercept=False)
    learner = SKLearner(lr)
    for i in range(4):
        learner.add(states[i], controls[i])
    learner.fit()
    print "Fitting score: " + str(learner.score())
    print "\nLearner: "
    print learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K"""
    






    print "\n\n\nLearner on 1 demonstration"
    lr = LinearRegression(fit_intercept=False)
    learner = SKLearner(lr)
    for i in range(1):
        learner.add(sup_states[i], sup_controls[i])
    learner.fit()
    print "Fitting score: " + str(learner.score())
    
    print "\nLearner: "
    print learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K

    trajs1 = []
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs1.append(states)



    print "\n\n\nLearner on 2 demonstration2"
    lr = LinearRegression(fit_intercept=False)
    learner = SKLearner(lr)
    for i in range(2):
        learner.add(sup_states[i], sup_controls[i])
    learner.fit()
    print "Fitting score: " + str(learner.score())
    
    print "\nLearner: "
    print learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K

    trajs2 = []
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs2.append(states)







    print "\n\n\nLearner on 3 demonstration"
    lr = LinearRegression(fit_intercept=False)
    learner = SKLearner(lr)
    for i in range(3):
        learner.add(sup_states[i], sup_controls[i])
    learner.fit()
    print "Fitting score: " + str(learner.score())
    
    print "\nLearner: "
    print learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K

    trajs3 = []    
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs3.append(states)
    
    


    print "\n\n\nLearner on 4 demonstration"
    lr = LinearRegression(fit_intercept=False)
    learner = SKLearner(lr)
    for i in range(4):
        learner.add(sup_states[i], sup_controls[i])
    learner.fit()
    print "Fitting score: " + str(learner.score())
    
    print "\nLearner: "
    print learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K

    trajs4 = []    
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs4.append(states)
    

    
    vis = Visualizer()
    vis.show_trajs(trajs1, x_f, "Trajs 1")
    vis.show_trajs(trajs2, x_f, "Trajs 2")
    vis.show_trajs(trajs3, x_f, "Trajs 3")
    vis.show_trajs(trajs4, x_f, "Trajs 4")



    """vis = Visualizer()
    vis.set_recording(states)
    vis.set_target(x_f)
    vis.show()

    print "Learner cost: " + str(sum(costs))
    """
    #print np.dot(learner.estimator.coef_, init_state)
    #print learner.predict(init_state)
    #print np.dot(robot.lqr.K, init_state - robot.x_f)

    


