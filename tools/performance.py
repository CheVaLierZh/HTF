import numpy as np
import psutil
import os
import time
from functools import wraps


# @njit()
def calculate_rmse(imputed, mask, data):
    """

    :param imputed: the data after imputed
    :param mask: the mask marked which element is missing, must be 0, 1 matrix, which 0 marks missing
    :param data: the origin data
    :return:
    """
    mask = np.array([mask == 0]).astype(int)
    return np.sqrt(np.sum(((imputed - data) * mask) ** 2) / np.sum(mask))


# @njit()
def calculate_nrmse(imputed, mask, data):
    """

    :param imputed: the data after imputed, same shape as data
    :param mask: the mask marked which element is missing in data, same shape as data
    :param data: the origin data
    :return:
    """
    mask = 1 - mask
    rmse = np.sqrt(np.sum(((imputed - data) * mask) ** 2) / np.sum(mask))
    nrmse = rmse / (np.sum(np.abs(data * mask)) / np.sum(mask))
    return nrmse


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
            (function.__name__, str(t1-t0))
            )
        return result
    return function_timer


def memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    X = np.random.randn(3, 3)
    m = np.random.randint(0, 2, (3, 3))
    print(calculate_nrmse(np.ones((3, 3)), m, X))