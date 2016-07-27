import numpy as np
ITERATIONS = 10
SAMPLES = 10

def compute_loss(states, controls, robot):
    loss = 0.0
    for state, control in zip(states, controls):
        sup_control = robot.pi(state, 0)
        loss += np.linalg.norm(control - sup_control) / float(len(controls))
    return loss


def compute_acc(robot, learner):
    diff = 0.0
    data = learner.data
    for state, control in data:
        pred_control = learner.predict(state)
        diff += np.linalg.norm(control - pred_control) / float(len(data))
    return diff



def supervisor_trial(robot, sys):
    trajs = []
    traj_controls = []
    total_costs = []
    avg_costs = np.zeros(ITERATIONS)
    for i in range(ITERATIONS):

        # Collect average costs over samples
        for s in range(SAMPLES):
            sys.reset_robot()
            states, controls, costs = robot.rollout(verbose=False)
            costs = np.array(costs) / float(SAMPLES)
            avg_costs[i] += sum(costs)

        # Learn and collect trajectories
        sys.reset_robot()
        states, controls, costs = robot.rollout(verbose=False)
        trajs.append(states)
        traj_controls.append(controls)
        total_costs.append(sum(costs))
    return trajs, traj_controls, total_costs, avg_costs



def supervise_trial(learner, robot, sys):
    # Assumes learner has no prior exploration
    trajs = []
    traj_controls = []
    total_costs = []
    avg_costs = np.zeros(ITERATIONS)
    avg_loss = np.zeros(ITERATIONS)
    for i in range(ITERATIONS):
        # Initial learning if no data
        if i == 0:
            state = robot.INITIAL_STATE
            control = robot.pi(state, 0)
            learner.add(state, control)
            learner.fit()
        

        # Collect average costs over samples with p_i
        for s in range(SAMPLES):
            sys.reset_robot()
            states, controls, costs = robot.rollout_learner(learner, verbose=False)
            costs = np.array(costs) / float(SAMPLES)
            avg_costs[i] += sum(costs)
            avg_loss[i] += compute_loss(states, controls, robot) / float(SAMPLES)

        # Learn and collect trajectories
        sys.reset_robot()
        sup_states, sup_controls, sup_costs = robot.rollout(verbose=False)
        for sup_state, sup_control in zip(sup_states, sup_controls):
            learner.add(sup_state, sup_control)
        learner.fit()

        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs.append(states)
        traj_controls.append(controls)
        total_costs.append(sum(costs))
    return trajs, traj_controls, total_costs, avg_costs, avg_loss


def dagger_trial(learner, robot, sys):
    # Assumes learner has no prior exploration
    trajs = []
    traj_controls = []
    total_costs = []
    avg_costs = np.zeros(ITERATIONS)
    avg_loss = np.zeros(ITERATIONS)
    for i in range(ITERATIONS):
        # Initial learning
        if i == 0:
            state = robot.INITIAL_STATE
            control = robot.pi(state, 0)
            learner.add(state, control)
            learner.fit()

        # Collect average costs
        for s in range(SAMPLES):
            sys.reset_robot()
            states, controls, costs = robot.rollout_learner(learner, verbose=False)
            costs = np.array(costs) / float(SAMPLES)
            avg_costs[i] += sum(costs)
            avg_loss[i] += compute_loss(states, controls, robot) / float(SAMPLES)


        # Learn and collect trajectories
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        for state in states:
            learner.add(state, robot.pi(state, 0))
        learner.fit()
        traj_controls.append(controls)
        trajs.append(states)
        total_costs.append(sum(costs))
    return trajs, traj_controls, total_costs, avg_costs, avg_loss


def dagger_final(learner, robot, sys):
    trajs = []
    traj_controls = []
    total_costs = []
    avg_costs = np.zeros(ITERATIONS)
    avg_loss = np.zeros(ITERATIONS)
    for i in range(ITERATIONS):
        # Collect average costs
        for s in range(SAMPLES):
            sys.reset_robot()
            states, controls, costs = robot.rollout_learner(learner, verbose=False)
            costs = np.array(costs) / float(SAMPLES)
            avg_costs[i] += sum(costs)
            avg_loss[i] += compute_loss(states, controls, robot) / float(SAMPLES)

        # Collect trajectories
        sys.reset_robot()
        states, controls, costs = robot.rollout_learner(learner, verbose=False)
        trajs.append(states)
        traj_controls.append(controls)
        total_costs.append(sum(costs))
    return trajs, traj_controls, total_costs, avg_costs, avg_loss









