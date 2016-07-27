from multisys import PiecewiseLinearSys
from env import highdim_multi as hdm
from vis import Visualizer
from multirobot import HighDimRobot 
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sk_learner import SKLearner
import numpy as np
import trials

if __name__ == '__main__':
    T = 200
    xdims = hdm.xdims
    udims = hdm.udims
    
    A1 = hdm.A1
    B1 = hdm.B1
    A2 = hdm.A2
    B2 = hdm.B2
    Q = hdm.Q
    R = hdm.R
    init_state = hdm.init_state
    
    x_f = np.zeros((xdims, 1))
    u_f =  np.zeros((udims, 1))

    sys = PiecewiseLinearSys(xdims, udims, T, A1, B1, A2, B2, stoch=True)
    robot = HighDimRobot(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    print "Performing LQR..."
    #robot.reg_lti()
    robot.lqr.loadKs()

    print "LQR Done."

    print "Rolling out..."
    sup_states, sup_controls, sup_costs = robot.rollout(verbose=True)
    print "Done rolling out."

    print "Supervised Learning"
    im_lr = LinearRegression()
    


    """
    data_directory = '/Users/JonathanLee/Documents/Research/lqr/data/'
    TRIALS = 1
    sup_data = np.zeros((TRIALS, trials.ITERATIONS))
    im_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_final_data = np.zeros((TRIALS, trials.ITERATIONS))

    im_loss_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_loss_data = np.zeros((TRIALS, trials.ITERATIONS))

    # SUPERVISOR
    print "Supervisor trajectories"
    for t in range(TRIALS):
        print "Supervisor trial: " + str(t)
        sup_trajs, sup_traj_controls, sup_costs, sup_avg_costs = trials.supervisor_trial(robot, sys)
        sup_data[t, :] = sup_avg_costs

    # SUPERVISED LEARNING
    print "Supervised Learning trajectories"
    for t in range(TRIALS):
        print "Supervised Learning trial: " + str(t)
        im_lr = LinearRegression(fit_intercept=False)#KernelRidge(kernel='rbf')
        im_learner = SKLearner(im_lr)
        im_trajs, im_traj_controls, im_costs, im_avg_costs, im_avg_loss = trials.supervise_trial(im_learner, robot, sys)
        im_data[t, :] = im_avg_costs
        im_loss_data[t, :] = im_avg_loss

    # DAGGER LEARNING
    print "DAgger Learning trajectories"
    for t in range(TRIALS):
        print "DAgger Learning trial: " + str(t)
        dagger_lr = LinearRegression(fit_intercept=False)#KernelRidge(kernel='rbf')
        dagger_learner = SKLearner(dagger_lr)
        dagger_trajs, dagger_traj_controls, dagger_costs, dagger_avg_costs, dagger_avg_loss = trials.dagger_trial(dagger_learner, robot, sys)
        dagger_final_trajs, dagger_final_controls, dagger_final_costs, dagger_final_avg_costs, dagger_final_avg_loss = trials.dagger_final(dagger_learner, robot, sys)

        dagger_data[t, :] = dagger_avg_costs
        dagger_loss_data[t, :] = dagger_avg_loss
        dagger_final_data[t, :] = dagger_final_avg_costs

    vis = Visualizer()
    #vis.show_trajs(sup_trajs, x_f, "sup_trajs", data_directory)
    #vis.show_trajs(im_trajs, x_f, "sl_trajs", data_directory)
    #vis.show_trajs(dagger_trajs, x_f, "dagger_trajs", data_directory)
    #vis.show_trajs(dagger_final_trajs, x_f, "dagger_final_trajs", data_directory)


    print robot.lqr.K1
    print im_lr.coef_
    print dagger_lr.coef_

    sup_costs = np.array(sup_data)
    im_costs = np.array(im_data)
    dagger_costs = np.array(dagger_data)

    print sup_costs
    print im_costs
    print dagger_costs

    filename = data_directory + 'cost_comparisons.eps'
    vis.get_perf(sup_costs)
    vis.get_perf(im_costs)
    vis.get_perf(dagger_costs)
    vis.plot(["Supervisor", "Supervised", "DAgger"], "Cost", filename)

    #filename = data_directory + 'surrogate_loss_comparisons.eps'
    #vis.get_perf(im_loss_data)
    #vis.get_perf(dagger_loss_data)
    #vis.plot(["Supervised", "DAgger"], "Loss", filename)
    """


    

"""


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
    print robot.lqr.K1

    trajs1 = []
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs1.append(states)

    


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
    print robot.lqr.K1

    trajs4 = []    
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs4.append(states)
    




    # DAGGER LEARNER
    # 
    # 
    #dlr = KernelRidge(kernel='rbf')
    dlr = LinearRegression(fit_intercept=False)
    dagger_learner = SKLearner(dlr)
    
    for i in range(1):
        dagger_learner.add(sup_states[i], sup_controls[i])
    dagger_learner.fit()
    print "Fitting score: " + str(dagger_learner.score())
    print "\nLearner: "
    #print dagger_learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K1

    trajs_dagger = []
    for i in range(20):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(dagger_learner, verbose=False)
        for state in states:
            dagger_learner.add(state, robot.pi(state, 0))
        dagger_learner.fit()
        trajs_dagger.append(states)

    print "Fitting score: " + str(dagger_learner.score())
    print "\nLearner: "
    #print dagger_learner.estimator.coef_
    print "\nLQR: "
    print robot.lqr.K1


    dagger_learner.fit()
    trajs_dagger_final = []
    for i in range(10):
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(dagger_learner, verbose=False)
        trajs_dagger_final.append(states)



    # 
    # 
    # END DAGGER LEARNER








    
    vis = Visualizer()
    vis.show_trajs(trajs1, x_f, "Trajs1")
    vis.show_trajs(trajs4, x_f, "Trajs4")
    vis.show_trajs(trajs_dagger, x_f, "TrasjsDAgger")
    vis.show_trajs(trajs_dagger_final, x_f, "TrajsFinalDAgger")


    """
