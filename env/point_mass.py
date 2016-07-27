import numpy as np
mass = float(20)
xdims = 4
udims = 2

A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0, 0], [0, 0], [1/mass, 0], [0, 1/mass]])
Q = np.identity(xdims)
R = np.identity(udims)
init_state = np.array([[-15], [-10], [0], [0]])

