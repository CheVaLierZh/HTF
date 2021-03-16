import numpy as np

from Algorithm.base import BaseRestorer
from Algorithm.TRMF.classes import TRMFRegressor
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from Algorithm.tools.performance import calculate_nrmse
from Algorithm.tools.performance import fn_timer
from Algorithm.tools.performance import memory_usage
import time


class TRMFRestorer(BaseRestorer):
    def __init__(self,
                 n_components,
                 n_order,
                 C_Z=1e-1,
                 C_F=1e-1,
                 C_phi=1e-2,
                 eta_Z=0.5,
                 eta_F=0.,
                 adj=None,
                 C_B=0.0,
                 fit_regression=False,
                 fit_intercept=True,
                 nonnegative_factors=True,
                 tol=1e-6,
                 n_max_iterations=1000,
                 n_max_mf_iter=5,
                 z_step_kind="tron",
                 f_step_kind="tron",
                 random_state=None
                 ):
        self.mem_start = memory_usage()
        self.n_max_iterations = n_max_iterations
        self.regressor = TRMFRegressor(n_components, n_order, C_Z=C_Z, C_F=C_F, C_phi=C_phi, eta_Z=eta_Z, eta_F=eta_F,
                                       adj=adj, C_B=C_B, fit_regression=fit_regression, fit_intercept=fit_intercept,
                                       nonnegative_factors=nonnegative_factors,
                                       tol=tol, n_max_iterations=n_max_iterations, n_max_mf_iter=n_max_mf_iter,
                                       z_step_kind=z_step_kind,
                                       f_step_kind=f_step_kind, random_state=random_state)

    def fit(self, X, W):
        def dense_matrix_to_sparse():
            rows_ind = []
            cols_ind = []
            data_val = []
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    if W[i, j] == 1:
                        rows_ind.append(i)
                        cols_ind.append(j)
                        data_val.append(X[i, j])

            smat = sparse.csr_matrix((data_val, (rows_ind, cols_ind)), shape=X.shape)
            return smat
        sps_data = dense_matrix_to_sparse()

        self.regressor.fit(sps_data)

    @fn_timer
    def restore(self, data, mask):
        data = data.T
        mask = mask.T

        t_start = time.time()
        self.fit(data, mask)
        t_end = time.time()


        recons = self.regressor.reconstruct()
        mem_cur = memory_usage()
        self.avg_time_cost = (t_end - t_start) / self.n_max_iterations
        self.mem_cost = mem_cur - self.mem_start
        print('Average time every iteration: %.2f s' % self.avg_time_cost)
        print('Consuming memory: %.2f MB' % self.mem_cost)
        return recons.T

    def restore_blackouts(self, data, mask):
        data = data.T
        mask = mask.T

        self.fit(data, mask)

        recons = self.regressor.reconstruct_blackout()
        return recons.T

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

        # print(get_mem())

        M = data.shape[0]
        N = data.shape[1]

        data[data_mask == 0] = np.nan

        restorer = TRMFRestorer(n_components=M-1, n_order=20, n_max_iterations=100, n_max_mf_iter=5)

        recovered = restorer.restore(data=data, mask=data_mask)

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
                # print(data[row, :])
                plt.plot(x, data_origin[row, :], color='tab:red', label='raw', ls=':')
                plt.plot(x, imputed, color='tab:blue', label='imputed', ls='-')
                plt.show()


    print('Average NRMSE: %f' % (rmse_avg / num_iter))
    print('Average time cost: %.2f' % (time_avg / num_iter))
