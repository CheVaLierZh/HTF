import numpy as np
from numba import njit


def _tkcm_impute(X, W, L, l, k):
    missing_pattern = X[:, L-l:L]
    D = np.zeros((L - 2 * l + 1,))
    for j in range(L - 2 * l + 1):
        D[j] = np.sqrt(np.sum((X[:, j:j+l] - missing_pattern) ** 2))

    M = np.zeros((k, L-2*l+1))
    for j in range(L-2*l+1):
        M[0, j] = 0
        for i in range(1, k, 1):
            if i > j:
                M[i, j] = np.inf
            else:
                M[i, j] = min(M[i, j-1], D[j] + M[i-1, max(j-l, 0)])

    i = k
    j = L-2*l+1
    A = np.zeros((k,)) - 1
    weights = np.zeros((k,))
    while i > 1 and j > 1:
        if M[i-1, j-1] == M[i-1, j-2]:
            j = j-1
        else:
            A[i-1] = j
            weights[i-1] = M[i-1, j-1] - M[i-2, max(j-l, 0)]
            i = i - 1
            j = max(j-l, 0)

    for i in range(k):
        if A[i] >= 0:
            break

    weights[i:] = weights[i:] / np.sum(weights[i:])
    weights[i:] = 1 - weights[i:]
    weights[i:] = weights[i:] / np.sum(weights[i:])
    weights[i:] = np.exp(weights[i:])
    weights[i:] = weights[i:] / np.sum(weights[i:])
    # print(weights[i:])

    k = k - i
    impute = np.zeros((X.shape[0],))
    for j in range(i, k, 1):
        impute += X[:, int(A[j]+l)] * weights[j]

    X[:, L][W[:, L] == 0] = impute[:][W[:, L] == 0]


def _tkcm(X, W, l, k):
    n_rows = X.shape[0]
    n_cols = X.shape[1]

    # impute all missing by columns one-by-one
    for i in range(n_cols):
        if np.sum(W[:, i]) != n_rows:
            _tkcm_impute(X, W, i, l, k)




