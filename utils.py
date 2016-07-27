import numpy as np

def matmul(*args):
    assert len(args) > 0
    product = args[0]
    for A in args[1:]:
        product = np.matmul(product, A)
    return product

def isPositive(x):
    for i in range(len(x)):
        if x[i, 0] <= 0:
            return False
    return True

#THESE WERE CHANGED, CHANGE THEM BACK
#

def inBoundary2(x):
    if x[0, 0] < 12 and x[1, 0] < 12:#12 and x[1, 0] < 12:#< -5 and x[1, 0] < -5:
        return True
    else:
        return False

def inBoundary(x):
    #border = -10e100
    #for i in range(len(x)):
    #    if x[i, 0] >= border:
    #        return False
    return True

