import numpy as np
from numba import njit


@njit()
def at(Z, i, j, k, M, L, K):
    return Z[i, j + k]


@njit()
def set_val(Z, i, j, k, M, L, K, val):
    Z[i, j + k] = val


class HankelTensor(object):
    def __init__(self, X, K):
        """

        :param X:
        :param K:
        """
        assert type(X) == np.ndarray
        self.X = X

        M, T = X.shape
        self.K = K
        self.L = T + 1 - self.K
        self.M = M

    def get_origin(self):
        return self.X

    def at(self, i, j, k):
        """

        :param i:
        :param j:
        :param k:
        :return:
        """
        assert i < self.M and j < self.L and k < self.K

        return self.X[i, j + k]
