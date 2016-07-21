import numpy as np

def matmul(*args):
    assert len(args) > 0
    product = args[0]
    for A in args[1:]:
        product = np.matmul(product, A)
    return product

