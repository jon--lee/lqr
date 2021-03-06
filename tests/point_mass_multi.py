from multisys import PiecewiseLinearSys
from env import point_mass_multi as pmm
#from env import mass_diff as md
from vis import Visualizer
from multirobot import MultiRobot 
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sk_learner import SKLearner
import numpy as np
import trials
import IPython

if __name__ == '__main__':
    T = 35
    xdims = pmm.xdims
    udims = pmm.udims
    
    A1 = pmm.A1
    B1 = pmm.B1
    A2 = pmm.A2#np.random.rand(4, 4)#pmm.A2
    B2 = pmm.B2#np.random.rand(4, 2)#pmm.B2
    Q = pmm.Q
    R = pmm.R
    init_state = pmm.init_state
    
    x_f = np.array([[0], [0], [0], [0]])#np.array([[0], [0], [0], [0]])#x_f = np.array([[10], [2], [0], [0]])
    u_f = np.array([[0], [0]])

    sys = PiecewiseLinearSys(xdims, udims, T, A1, B1, A2, B2, stoch=True)
    robot = MultiRobot(sys, init_state, T, Q, R, x_f=x_f, u_f=u_f)
    robot.reg_lti()

    #robot.lqr.K2 = np.random.rand(2, 4)

    data_directory = '/Users/JonathanLee/Documents/Research/lqr/data/'
    TRIALS = 20
    sup_data = np.zeros((TRIALS, trials.ITERATIONS))
    im_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_final_data = np.zeros((TRIALS, trials.ITERATIONS))

    im_loss_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_loss_data = np.zeros((TRIALS, trials.ITERATIONS))

    im_acc_data = np.zeros((TRIALS, trials.ITERATIONS))
    dagger_acc_data = np.zeros((TRIALS, trials.ITERATIONS))

    # SUPERVISOR
    print "Supervisor trajectories"
    for t in range(TRIALS):
        print "Supervisor trial: " + str(t)
        sup_trajs, sup_traj_controls, sup_costs, sup_avg_costs = trials.supervisor_trial(robot, sys)
        sup_data[t, :] = sup_avg_costs

    # SUPERVISED LEARNIN
    print "Supervised Learning trajectories"
    for t in range(TRIALS):
        print "Supervised Learning trial: " + str(t)
        # im_lr = KernelRidge(kernel='poly', gamma=100, degree=3)
        # im_lr = KernelRidge(kernel='rbf', gamma=100)
        im_lr = LinearRegression(fit_intercept=False)#KernelRidge(kernel='rbf')#
        im_learner = SKLearner(im_lr)
        im_trajs, im_traj_controls, im_costs, im_avg_costs, im_avg_loss, im_accs = trials.supervise_trial(im_learner, robot, sys)
        im_data[t, :] = im_avg_costs
        im_loss_data[t, :] = im_avg_loss
        im_acc_data[t, :] = im_accs

    # DAGGER LEARNING
    print "DAgger Learning trajectories"
    for t in range(TRIALS):
        print "DAgger Learning trial: " + str(t)
        # dagger_lr = KernelRidge(kernel='poly', gamma=100, degree=3)
        # dagger_lr = KernelRidge(kernel='rbf', gamma=100)
        dagger_lr = LinearRegression(fit_intercept=False)#KernelRidge(kernel='rbf')#
        dagger_learner = SKLearner(dagger_lr)
        dagger_trajs, dagger_traj_controls, dagger_costs, dagger_avg_costs, dagger_avg_loss, dagger_accs = trials.dagger_trial(dagger_learner, robot, sys)
        dagger_final_trajs, dagger_final_controls, dagger_final_costs, dagger_final_avg_costs, dagger_final_avg_loss = trials.dagger_final(dagger_learner, robot, sys)

        dagger_data[t, :] = dagger_avg_costs
        dagger_loss_data[t, :] = dagger_avg_loss
        dagger_final_data[t, :] = dagger_final_avg_costs
        dagger_acc_data[t, :] = dagger_accs

    # print im_lr.coef_
    # print dagger_lr.coef_
    # print robot.lqr.K1

    vis = Visualizer()
    vis.show_trajs(sup_trajs, x_f, "sup_trajs", data_directory)
    vis.show_trajs(im_trajs, x_f, "sl_trajs", data_directory)
    vis.show_trajs(dagger_trajs, x_f, "dagger_trajs", data_directory)
    vis.show_trajs(dagger_final_trajs, x_f, "dagger_final_trajs", data_directory)



    print im_accs
    print dagger_accs

    print "\n\n\n"
    for state, control in im_learner.data:
        print control
        print im_learner.predict(state)
        print dagger_learner.predict(state)
        state = state.reshape((xdims, 1))
        print robot.pi(state, 0)
        
        break
        
    print "\n\n\n"


    print trials.compute_acc(im_learner)
    print trials.compute_acc(dagger_learner)


    sup_costs = np.array(sup_data)
    im_costs = np.array(im_data)
    dagger_costs = np.array(dagger_data)

    print "\n\n\nCOSTS"
    print "Sup costs"
    print sup_costs
    print "SL costs"
    print im_costs
    print "Dagger costs"
    print dagger_costs
    print "\n\n\n"
    im_accs = np.array([im_accs])
    dagger_accs = np.array([dagger_accs])


    data_dir = './data/raw/'
    np.save(data_dir + 'sup_costs', sup_costs)
    np.save(data_dir + 'im_costs', im_costs)
    np.save(data_dir + 'dagger_costs', dagger_costs)

    filename = data_directory + 'cost_comparisons.eps'
    vis.get_perf(sup_costs)
    vis.get_perf(im_costs)
    vis.get_perf(dagger_costs)
    vis.plot(["Supervisor", "Supervised", "DAgger"], "Cost", filename)

    filename = data_directory + 'surrogate_loss_comparisons.eps'
    vis.get_perf(im_loss_data)
    vis.get_perf(dagger_loss_data)
    vis.plot(["Supervised", "DAgger"], "Loss", filename)



    filename = data_directory + 'accuracy_comparisons.eps'
    vis.get_perf(im_acc_data)
    vis.get_perf(dagger_acc_data)
    vis.plot(['Supervised', 'DAgger'], "Acc", filename)


    """
    print "\nGrid searching"
    params = {"gamma": [.01, .05, .1, .5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        }#"degree": [1, 2, 3, 4, 5, 6]}

    im_best_est, im_best_params = im_learner.gridsearch(params)
    dagger_best_est, dagger_best_params = dagger_learner.gridsearch(params)
    print "\nBest parameters"
    print im_best_params
    print dagger_best_params

    im_learner.estimator = im_best_est
    dagger_learner.estimator = dagger_best_est

    print "\nResulting training losses on gridsearched params"
    print trials.compute_acc(im_learner)
    print trials.compute_acc(dagger_learner)
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
