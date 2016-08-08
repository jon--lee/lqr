import numpy as np
mass = float(20)
mass2 = float(1.5)
xdims = 4
udims = 2

A1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
B1 = np.array([[0, 0], [0, 0], [1/mass, 0], [0, 1/mass]])
A2 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
# B2 = np.array([[0, 0], [0, 0], [0, 1/mass], [1/mass, 0]])
B2 = np.array([[0, 0], [0, 0], [1/mass2, 0], [0, 1/mass2]])

Q = np.identity(xdims)
R = np.identity(udims)
init_state = np.array([[-10], [-15], [0], [0]])


