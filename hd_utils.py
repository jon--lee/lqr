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


def inBoundary(x):
    #border = -10e100
    #for i in range(len(x)):
    #    if x[i, 0] >= border:
    #        return False
    return True

