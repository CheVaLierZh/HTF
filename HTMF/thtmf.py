import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from numba.errors import NumbaPerformanceWarning

import Algorithm.HTMF.hankel_tensor as hankel_tensor
from Algorithm.HTMF.hankel_tensor import HankelTensor
from Algorithm.base import BaseRestorer
from Algorithm.tools.performance import calculate_nrmse, memory_usage

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


class THTMFRestorer(BaseRestorer):
    """
        This method is used to recover multivariate time series
        it composite these following steps:
            1. hankelization: X = [[x11,...,x1n],...,[xm1,...,xmn]] =>
                                [[[x11,...,x1l],...,[xm1,...,xml]],...,
                                [[x1k,...,x1n],...,[xmk,...,xmn]]]
            2. H_{M*L*K}(X): tensor factorization
            3. diagonal averaging for missing value:
    """

    def __init__(self, iter_num=100, calculate_training_loss=True, lambda_s=1e-4, lambda_re=1e-4, eta=1e-3, xi=1.2):

        self.mem_start = memory_usage()
        self.iter_num = iter_num
        self.calculate_training_loss = calculate_training_loss
        self.lambda_s = lambda_s
        self.lambda_re = lambda_re
        self.xi = xi
        self.eta = eta

        self.A = None
        self.B = None
        self.C = None
        # self.E = None

        self.HX = None

        self.M = 0
        self.L = 0
        self.K = 0
        self.R = 0

        self.loss_iter = None

        self.time_cost = 0
        self.real_iters = 0

    def fit(self, X, W):
        assert X.shape == W.shape and np.ndim(X) == 2

        X[W == 0] = np.nan

        def _find_max_continuous_missing(_W):
            ret = 0
            _M = W.shape[0]
            _N = W.shape[1]
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

        K = _find_max_continuous_missing(W)

        self.HX = HankelTensor(X, int(self.xi * K))

        self.R = X.shape[0] + 20

        self.M, self.L, self.K = self.HX.M, self.HX.L, self.HX.K

        self.A = np.random.randn(self.M, self.R) / np.sqrt(np.max((self.M, self.R)))
        self.B = np.random.randn(self.L, self.R) / np.sqrt(np.max((self.L, self.R)))
        self.C = np.random.randn(self.K, self.R) / np.sqrt(np.max((self.K, self.R)))
        # self.E = HankelTensor(np.zeros(X.shape), self.HX.K)

        self.loss_iter = []
        for i in range(self.iter_num):
            t_start = time.time()
            print('Learning rate: %f' % (self.eta))
            self.solver()
            t_end = time.time()
            self.time_cost += t_end - t_start
            self.real_iters = i + 1
            print('consuming time: %.2f s' % (t_end - t_start))
            if self.calculate_training_loss:
                loss = self.training_loss()
                print('Iteration %d training loss: %.2f' % (i + 1, loss))
                if i > 5:
                    if all([self.loss_iter[j] < self.loss_iter[j + 1] for j in range(-5, -1)]):
                        self.iter_num = i + 1
                        self.loss_iter.append(loss)
                        break
                if i > 0:
                    self._update_step(loss <= self.loss_iter[-1], i + 1)
                self.loss_iter.append(loss)

    def solver(self):

        @njit()
        def sgd(X, A, B, C, lambda_s, lambda_re, eta, M, L, K):
            for m in range(M):
                for l in range(L):
                    for k in range(K):
                        H_val = hankel_tensor.at(X, m, l, k, M, L, K)
                        if np.isnan(H_val):
                            continue

                        e_mlk = H_val - np.dot(A[m, :], np.multiply(B[l, :], C[k, :]))
                        gradientAm = e_mlk * (-np.multiply(B[l, :], C[k, :])) + lambda_re * A[m, :]
                        gradientBl = e_mlk * (-np.multiply(A[m, :], C[k, :])) + lambda_re * B[l, :]
                        gradientCk = e_mlk * (-np.multiply(A[m, :], B[l, :])) + lambda_re * C[k, :]

                        """
                        matAmCk = np.multiply(A[m, :], C[k, :])
                        matAmBl = np.multiply(A[m, :], B[l, :])
                        matBlCk = np.multiply(B[l, :], C[k, :])
                        hessianAm = np.dot(np.transpose(matBlCk),
                                           matBlCk) + lambda_re * np.eye(R)
                        hessianAm = np.linalg.inv(hessianAm)
                        hessianBl = np.dot(np.transpose(matAmCk),
                                           matAmCk) + lambda_re * np.eye(R)
                        hessianBl = np.linalg.inv(hessianBl)
                        # print(hessianBl[:4, :4])
                        hessianCk = np.dot(np.transpose(matAmBl),
                                           matAmBl) + lambda_re * np.eye(R)
                        hessianCk = np.linalg.inv(hessianCk)
                        """
                        if k % 10 == 0:
                            if 0 < l:
                                tmpBl = B[l, :] - B[l - 1, :]
                                continuousl = np.dot(A[m, :], np.multiply(tmpBl, C[k, :]))

                                gradientAm += lambda_s * np.multiply(continuousl, np.multiply(tmpBl, C[k, :]))

                                gradientBl += lambda_s * np.multiply(continuousl, np.multiply(A[m, :], C[k, :]))

                                gradientCk += lambda_s * np.multiply(continuousl, np.multiply(A[m, :], tmpBl))

                            if l < L - 1:
                                tmpBl_1 = B[l + 1, :] - B[l, :]
                                continuousl_1 = np.dot(A[m, :], np.multiply(tmpBl_1, C[k, :]))

                                gradientAm += lambda_s * np.multiply(continuousl_1, np.multiply(tmpBl_1, C[k, :]))

                                gradientBl += -lambda_s * np.multiply(continuousl_1, np.multiply(A[m, :], C[k, :]))

                                gradientCk += np.multiply(continuousl_1, np.multiply(A[m, :], tmpBl_1))

                        """
                        A[m, :] -= eta * np.dot(gradientAm, hessianAm)
                        B[l, :] -= eta * np.dot(gradientBl, hessianBl)
                        C[k, :] -= eta * np.dot(gradientCk, hessianCk)
                        """
                        A[m, :] -= eta * gradientAm
                        B[l, :] -= eta * gradientBl
                        C[k, :] -= eta * gradientCk

        return sgd(self.HX.get_origin(), self.A, self.B, self.C, self.lambda_s, self.lambda_re,
                   self.eta, self.HX.M, self.HX.L, self.HX.K)

    def training_loss(self):

        @jit(nopython=True)
        def calculate(H, A, B, C, lambda_s, lambda_re, M, L, K):
            loss = 0.0
            for m in range(M):
                for l in range(L):
                    for k in range(K):
                        H_val = hankel_tensor.at(H, m, l, k, M, L, K)
                        # E_val = hankel_tensor.at(H, m, l, k, M, L, K)
                        if np.isnan(H_val):
                            continue
                        loss += (np.dot(A[m, :], np.multiply(B[l, :], C[k, :])) - H_val) ** 2
                        if l > 0:
                            tmpBl = B[l, :] - B[l - 1, :]
                            loss += lambda_s * (np.dot(A[m, :], np.multiply(tmpBl, C[k, :]))) ** 2

            loss += lambda_re * (np.sum(A ** 2) + np.sum(B ** 2) + np.sum(C ** 2))
            return loss

        return calculate(self.HX.get_origin(), self.A, self.B, self.C, self.lambda_s,
                         self.lambda_re,
                         self.HX.M, self.HX.L, self.HX.K)

    def _update_step(self, flag, i):
        if flag:
            self.eta = np.min((self.eta * (1 + 2 * np.exp(-np.log10(i))), 5e-2))
        else:
            self.eta = self.eta / 10

    def diag_average_impute(self, X, W):
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                    sum = 0.0
                    cnt = 0
                    for l in range(self.L):
                        k = n - l
                        if 0 <= k < self.K:
                            cnt += 1
                            sum += np.dot(self.A[m, :], np.multiply(self.B[l, :], self.C[k, :]))
                    X[m, n] = sum / cnt
        return X

    def restore(self, data, mask):
        self.fit(data, mask)
        ret =  self.diag_average_impute(data, mask)
        mem_cur = memory_usage()
        self.avg_time_cost = self.time_cost / self.real_iters
        self.mem_cost = mem_cur - self.mem_start
        print('Average time every iteration: %.2f s' % (self.time_cost / self.real_iters))
        print('Consuming memory: %.2f MB' % (mem_cur - self.mem_start))
        return ret

    def get_time_space(self):
        return self.avg_time_cost, self.mem_cost


if __name__ == '__main__':
    rmse_avg = 0.0
    num_iter = 3

    time_avg = 0.0
    mem_avg = 0.0

    data_file = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Electricity\\electricity_normal.txt'
    for _ in range(num_iter):
        data_mask_file = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Electricity\\' \
                         'electricity_normal_blackout5_' + str(_+1) + '.txt'
        me1 = memory_usage()
        data_origin = np.loadtxt(data_file, delimiter=' ', dtype=np.float)
        data_mask = np.loadtxt(data_mask_file, delimiter=' ', dtype=np.int)
        data = data_origin.copy()
        me2 = memory_usage()

        M = data.shape[0]
        N = data.shape[1]

        data[data_mask == 0] = np.nan

        restorer = THTMFRestorer(iter_num=100, lambda_re=1e-4, lambda_s=1e-4, xi=1.1)

        recovered = restorer.restore(data, data_mask)
        a, b = restorer.get_time_space()
        time_avg += a
        mem_avg += b

        rmse = calculate_nrmse(imputed=recovered, mask=data_mask, data=data_origin)
        rmse_avg += rmse
        print('NRMSE: %f' % rmse)


        for row in range(M):
            if np.sum(data_mask[row, :]) != N:
                mask_row_0 = data_mask[row, :]
                imputed = recovered[row, :]
                imputed[mask_row_0 == 1] = np.nan
                x = np.arange(data_origin.shape[1])
                plt.plot(x, data_origin[row, :], color='tab:red', label='raw', ls=':')
                plt.plot(x, imputed, color='tab:blue', label='imputed', ls='-')
                plt.show()


    print('Average NRMSE is %f' % (rmse_avg / num_iter))
    print('Average time cost: %.2f' % (time_avg / num_iter))
    print('Average memory cost: %.2f' % (mem_avg / num_iter))