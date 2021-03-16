import numpy as np


def _find_max_continuous_missing(_W, axis=0):
    if axis == 0:
        _W = _W.T
    ret = 0
    _M = _W.shape[0]
    _N = _W.shape[1]
    for _i in range(_M):
        start = _N
        for _j in range(_N):
            if _W[_i, _j] == 0 and start == _N:
                start = _j
            elif _W[_i, _j] == 1 and start != _N:
                ret = np.max((ret, _j - start))
                start = _N
                continue
            if _j == _N - 1:
                ret = np.max((ret, _N - start))
    return ret
