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


def inBoundary(x):
    #return  True
    border = 12#-5#12
    if x[0, 0] < border and x[1, 0] < border:#12 and x[1, 0] < 12:#< -5 and x[1, 0] < -5:
        return True
    else:
        return False
    # border = -12
    # if x[0, 0] > border:
    #     return False
    # else:
    #     return True

def inBoundary2(x):
    #border = -10e100
    #for i in range(len(x)):
    #    if x[i, 0] >= border:
    #        return False
    return True

