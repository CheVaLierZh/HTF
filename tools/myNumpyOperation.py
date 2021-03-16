import numpy as np
from numba import njit


@njit()
def my_roll_axis1(X, l, K, T):
    Y = np.zeros((K, T))
    idx = -l
    for i in range(T):
        Y[:, i] = X[:, idx]
        idx = idx + 1
        if idx >= T:
            idx = 0
    return Y
