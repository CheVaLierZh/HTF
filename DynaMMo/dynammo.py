import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from Algorithm.base import BaseRestorer
from Algorithm.tools.performance import calculate_nrmse
from Algorithm.tools.performance import fn_timer
from Algorithm.tools.performance import memory_usage
import time


class DynammoRestorer(BaseRestorer):
    """
        The implementation of paper 'Dynammo: mining and summarization of coevolvingsequences with missing values'
        Linear Dynamic System:
            Zn = AZn-1 + Q
            Xn = CXn-1 + R
            Z1 = Z0 + N(0, Q0)

        Parameters:
        -----------------
        X: shape(m_variates, n_timeSeries)
            the raw multivariate time series
        observed: shape(m_variates, n_timeSeries)
            the mask vector represent which element in X_original is missing or not
        M: int
            the number of multivariate in X_original
        N: int
            the length of time series
        H: int
            the dimensions of hidden variate
        max_iter: int
            the max number of iterations MLE will run
        A: the transition matrix between hidden variables
        C: the transition matrix between hidden variables to observed variables
        conv_bound: converage mark: logli - old_logli <= conv_bound => converage,
            logli, old_logli: E_{X_m, Z|X_g, theta}[P(X_g, Z)] this iteration, previous iteration

    """

    def __init__(self, max_iter=100, conv_bound=1e-5):
        super(BaseRestorer, self).__init__()
        self.mem_start = memory_usage()

        self.M = None
        self.N = None
        self.H = None

        self.iter = 0
        self.max_iter = max_iter
        self.conv_bound = conv_bound

        self.A = None
        self.C = None
        self.Q = None
        self.R = None
        self.mu0 = None
        self.Q0 = None

        self.logli_per_iter = None

        self.time_cost = 0
        self.real_iters = 0

    def forward_process(self, X):
        """
        do Kalman Filter forward process

        :param X: n_variable, n_timepoints: the observed time series
        :return:
        """
        mu = np.zeros((self.N, self.H, 1))  # mu[t] = E[zt|x1,...,xt]
        V = np.zeros((self.N, self.H, self.H))  # V[t] = Var[zt|x1,...,xt]
        P = np.zeros((self.N, self.H, self.H))  # P[t] = Var[zt|x1,...,xt-1]

        logli = 0.0  # logli = logP(x1,...,xn) = sigma{logP(xt|x1,...,xt-1)}, P(xt|x1,...,xt-1)~N(mu,s)

        for t in range(self.N):
            if t == 0:
                mu[t] = self.mu0
                P[t] = self.Q0
            else:
                P[t] = self.A.dot(V[t-1]).dot(self.A.T) + self.Q
                mu[t] = self.A.dot(mu[t-1])
            Pt = P[t]
            sigma_c = self.C.dot(Pt).dot(self.C.T) + self.R
            inv_sig = np.linalg.inv(sigma_c)
            K = Pt.dot(self.C.T).dot(inv_sig)
            delta = X[:, t].reshape(-1, 1) - self.C.dot(mu[t])
            mu[t] = mu[t] + K.dot(delta)
            V[t] = (np.eye(self.H, self.H) - K.dot(self.C)).dot(Pt)

            posDef = delta.T.dot(inv_sig).dot(delta) / 2
            logli = logli - self.M / 2 * np.log(2*np.pi) + np.log(np.linalg.det(inv_sig)) / 2 - posDef
            logli = float(logli)
        return mu, V, P, logli

    def backward_process(self, mu, V, P):
        """
        do Kalman Filter backward process

        :param mu: expectation vec from forward process
        :param V: Variance vec from forward process
        :param P: from forward process
        :return:
        """
        Ez = np.zeros((self.N, self.H, 1))  # Ez[t] = E[zt|x1,...,xn]
        Ezz = np.zeros((self.N, self.H, self.H))  # Ezz[t] = E[ztzt'|x1,...,xn]
        Ezz1 = np.zeros((self.N, self.H, self.H))  # Ezz1[t] = E[ztzt-1|x1,...,xn]

        Ez[self.N - 1] = mu[self.N - 1]
        Vhat = V[self.N - 1]
        Ezz[self.N - 1] = Vhat + Ez[self.N - 1].dot(Ez[self.N - 1].T)

        for t in range(self.N - 1, 0, -1):
            J = V[t - 1].dot(self.A.T).dot(np.linalg.inv(P[t]))  # Jt-1
            Ez[t - 1] = mu[t - 1] + J.dot(Ez[t] - self.A.dot(mu[t - 1]))  # zt-1|x1,...,xn
            Ezz1[t] = J.dot(Vhat) + Ez[t].dot(Ez[t - 1].T)
            Vhat = V[t - 1] + J.dot(Vhat - P[t]).dot(J.T)  # Vt-1|x1,...,xn
            Ezz[t - 1] = Vhat + Ez[t - 1].dot(Ez[t - 1].T)

        return Ez, Ezz, Ezz1

    def MLE_lds(self, X, Ez, Ezz, Ezz1):
        """
        do Expectation Maximization process of Kalman Filter

        :param X:
        :param Ez:
        :param Ezz:
        :param Ezz1:
        :return:
        """
        Szz1 = np.zeros((self.H, self.H))  # sum(Eztzt-1) from 2 to N
        Szz = np.zeros((self.H, self.H))  # sum(Ezz) from 1 to N
        Sxz = np.zeros((self.M, self.H))  # sum(xtEzt) from 1 to N

        for t in range(1, self.N):
            Szz1 = Szz1 + Ezz1[t]

        for t in range(self.N):
            Szz = Szz + Ezz[t]
            Sxz = Sxz + X[:, t].reshape(-1, 1).dot(Ez[t].T)

        SzzN = Szz - Ezz[self.N - 1]
        self.mu0 = Ez[0]
        self.Q0 = Ezz[0] - Ez[0].dot(Ez[0].T)

        self.A = Szz1.dot(np.linalg.inv(SzzN))

        tmp = Szz1.dot(self.A.T)
        self.Q = (Szz - Ezz[0] - tmp.T - tmp + self.A.dot(SzzN).dot(self.A.T)) / (self.N - 1)
        self.C = Sxz.dot(np.linalg.inv(Szz))

        tmp = self.C.dot(Sxz.T)
        self.R = (X.dot(X.T) - tmp - tmp.T + self.C.dot(Szz).dot(self.C.T)) / self.N

    def estimate_missing(self, X, Ez):
        """
        get the estimation of kalman filter

        :param X: n_variable, n_timepoints: observed time series
        :param Ez: expectation of hidden variables
        :return:
        """
        Y = np.zeros(X.shape)
        for t in range(self.N):
            Y[:, t] = self.C.dot(Ez[t]).reshape(-1, )
        return Y

    @fn_timer
    def restore(self, data, mask):
        assert data.shape == mask.shape

        self.M = data.shape[0]
        self.N = data.shape[1]

        self.A = np.eye(self.H, self.H) + np.random.randn(self.H, self.H)
        self.C = np.eye(self.M, self.H) + np.random.randn(self.M, self.H)
        self.Q = np.eye(self.H, self.H)
        self.R = np.eye(self.M, self.M)
        self.mu0 = np.random.randn(self.H, 1)
        self.Q0 = self.Q

        # implementation of linear interp specified for this
        def linear_interp(X, W):
            for i in range(W.shape[0]):
                obs = np.nonzero(W[i, :])[0]
                unobs = np.nonzero(W[i, :] - 1)[0]
                if obs.shape[0] > 1:
                    f = interpolate.interp1d(obs, X[i, obs], 'linear', fill_value='extrapolate')
                    X[i, unobs] = f(unobs)

        # first impute missing values with interpolation methods
        linear_interp(data, mask)

        diff = 1  # logli - old_logli
        old_logli = float('-inf')

        self.iter = 0
        self.logli_per_iter = []
        while diff > self.conv_bound and self.iter < self.max_iter:
            self.iter = self.iter + 1
            t_start = time.time()
            mu, V, P, logli = self.forward_process(data)
            Ez, Ezz, Ezz1 = self.backward_process(mu, V, P)
            self.MLE_lds(data, Ez, Ezz, Ezz1)
            data_estimate = self.estimate_missing(data, Ez)
            data[mask == 0] = data_estimate[mask == 0]
            t_end = time.time()
            print('consuming time: %.2f s' % (t_end - t_start))
            self.time_cost += t_end - t_start
            self.real_iters += 1
            diff = logli - old_logli
            self.logli_per_iter.append(logli)
            old_logli = logli
            print('Iteration %d log likelihood: %.4f' % (self.iter, logli))

        data = data_estimate

        mem_cur = memory_usage()
        self.avg_time_cost = self.time_cost / self.real_iters
        self.mem_cost = mem_cur - self.mem_start
        print('Average time every iteration: %.2f s' % (self.time_cost / self.real_iters))
        print('Consuming memory: %.2f MB' % (mem_cur - self.mem_start))
        return data

    def setHiddenDimension(self, H):
        """
        set the number of hidden variables

        :param H: int
        :return:
        """
        self.H = H

    def get_time_space(self):
        return self.avg_time_cost, self.mem_cost


if __name__ == '__main__':
    rmse_avg = 0.0
    num_iter = 5

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

        M = data_origin.shape[0]
        N = data_origin.shape[1]

        data[data_mask == 0.0] = 0

        restorer = DynammoRestorer(max_iter=100, conv_bound=1e-4)
        restorer.setHiddenDimension(M // 2)
        restored = restorer.restore(data=data, mask=data_mask)
        # restorer.plot_log_likelihood()

        a, b = restorer.get_time_space()
        time_avg += a
        mem_avg += b

        # calculate rmse
        rmse = calculate_nrmse(imputed=restored, mask=data_mask, data=data_origin)
        rmse_avg += rmse
        print('NRMSE: %f' % rmse)


        for row in range(M):
            if np.sum(data_mask[row, :]) != N:
                mask_row_0 = data_mask[row, :]
                imputed = restored[row, :]
                imputed[mask_row_0 == 1] = np.nan
                x = np.arange(N)
                # print(data[row, :])
                plt.plot(x, data_origin[row, :], color='tab:red', label='raw', ls=':')
                plt.plot(x, imputed, color='tab:blue', label='imputed', ls='-')
                plt.show()


    print('Average NRMSE: %f' % (rmse_avg / num_iter))
    print('Average time cost: %.2f' % (time_avg / num_iter))