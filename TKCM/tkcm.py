import numpy as np
from Algorithm.base import BaseRestorer
from Algorithm.TKCM.base import _tkcm
from Algorithm.tools.performance import calculate_nrmse
import matplotlib.pyplot as plt
from Algorithm.tools.performance import fn_timer
from Algorithm.tools.performance import memory_usage
import time


class TKCMRestorer(BaseRestorer):
    def __init__(self, l=5, k=3):
        self.l = l
        self.k = k

    @fn_timer
    def restore(self, data, mask):
        for i in range(data.shape[1]):
            if np.sum(mask[:, i]) != data.shape[0]:
                break
        if i < 2 * self.l:
            if i > 1:
                print('setting pattern length %d not matching missing, reset it to %d' % (self.l, i))
                self.l = i
            else:
                raise Exception('Error when checking pattern length, failed because of the first missing is '
                      'in the first or second columns')

        t_start = time.time()
        mem_start = memory_usage()
        _tkcm(data, mask, self.l, self.k)
        t_end = time.time()
        mem_cur = memory_usage()

        self.avg_time_cost = t_end - t_start
        self.mem_cost = mem_cur - mem_start
        print('Consuming time: %.2f s' % (t_end - t_start))
        print('Consuming memory: %.2f MB' % (mem_cur - mem_start))


        return data

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

        # print(get_mem())

        M = data.shape[0]
        N = data.shape[1]

        data[data_mask == 0] = np.nan

        restorer = TKCMRestorer(l=35, k=5)

        recovered = restorer.restore(data=data, mask=data_mask)
        a, b = restorer.get_time_space()
        time_avg += a
        mem_avg += b

        rmse = calculate_nrmse(imputed=recovered, mask=data_mask, data=data_origin)
        # restorer.save_params('D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Air Quality')
        rmse_avg += rmse
        print('NRMSE: %f' % rmse)


        for row in range(M):
            if np.sum(data_mask[row, :]) != N:
                mask_row_0 = data_mask[row, :]
                imputed = recovered[row, :]
                imputed[mask_row_0 == 1] = np.nan
                x = np.arange(data_origin.shape[1])
                # print(data[row, :])
                plt.plot(x, data_origin[row, :], color='tab:red', label='raw',ls=':')
                plt.plot(x, imputed, color='tab:blue', label='imputed', ls='-')
                plt.show()


    print('Average NRMSE: %f' % (rmse_avg / num_iter))
    print('Average time cost: %.2f' % (time_avg / num_iter))
    print('Average memory cost: %.2f' % (mem_avg / num_iter + me2 - me1))

