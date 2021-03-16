from Algorithm.BRITS.BRITS_Learner import BRITSLearner
from Algorithm.tools.TENsor.Preprocess_Data import PieceDataPreprocessor
import argparse
from Algorithm.tools.TENsor.Metrics import rmse
import os
import numpy as np
from Algorithm.tools.performance import calculate_nrmse
import matplotlib.pyplot as plt
from Algorithm.tools.performance import memory_usage


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--delim',
        default=' ',
        type=str
    )

    parser.add_argument(
        '--row_is_time',
        default=False,
        type=bool
    )

    parser.add_argument(
        '--skip_row',
        default=0,
        type=int
    )

    parser.add_argument(
        '--seq_len',
        default=150,
        type=int
    )

    parser.add_argument(
        '--batch_size',
        default=32,
        type=int
    )

    parser.add_argument(
        '--scale_method',
        default=None,
        type=str
    )

    parser.add_argument(
        '--learning_rate',
        default=1e-2,
        type=float
    )

    parser.add_argument(
        '--optim_method',
        default='Adam',
        type=str
    )

    parser.add_argument(
        '--max-epoch',
        default=10,
        type=int
    )

    parser.add_argument(
        '--hidden_dim',
        default=64,
        type=int
    )

    args = parser.parse_args()

    rmse_avg = 0.0
    num_iter = 5

    time_avg = 0.0
    mem_avg = 0.0


    datafile2 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Chlorine\\chlorine_normal.txt'
    maskfile2 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Chlorine\\' \
                          'chlorine_normal_blackout5_' + '3' + '.txt'

    datafile3 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Electricity\\electricity_normal.txt'
    maskfile3 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Electricity\\' \
                'electricity_normal_blackout5_' + '1' + '.txt'

    datafile4 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Gas\\gas_normal.txt'
    maskfile4 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Gas\\' \
                'gas_normal_overlap' + '1' + '.txt'

    datafile5 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Temperature\\temp_normal.txt'
    maskfile5 = 'D:\\Graduation Project\\Time_Series_Restoration\\Dataset\\Temperature\\' \
                'temp_normal_blackout5_' + '3' + '.txt'
    for _ in range(num_iter):
        _ = 2
        if _ == 0:
            continue
        elif _ == 1:
            datafile = datafile2
            maskfile = maskfile2
        elif _ == 2:
            datafile = datafile3
            maskfile = maskfile3
        elif _ == 3:
            datafile = datafile4
            maskfile = maskfile4
            args.hidden_dim = 128
        elif _ == 4:
            args.hidden_dim = 64
            datafile = datafile5
            maskfile = maskfile5

        preprocessor = PieceDataPreprocessor(args.seq_len, args.batch_size, args.scale_method)

        data, mask = preprocessor.readfile(datafile, maskfile, args.delim, args.row_is_time, args.skip_row)

        learner = BRITSLearner(optim_method=args.optim_method, max_epoch=args.max_epoch, lr=args.learning_rate)
        learner.accept_data_preprocessor(preprocessor)

        learner.set_hidden_dim(args.hidden_dim)
        imputed, data_m, mask_m = learner.restore(data, mask)

        a, b = learner.get_time_space()
        time_avg += a
        mem_avg += b
        print('NRMSE: %.2f' % calculate_nrmse(imputed, mask_m, data_m))

        for col in range(data_m.shape[1]):
            if np.sum(mask_m[:, col]) != data_m.shape[0]:
                mask_col_0 = mask_m[:, col]
                imputed_col = imputed[:, col]
                imputed_col[mask_col_0 == 1] = np.nan
                x = np.arange(data_m.shape[0])
                # print(data[row, :])
                plt.plot(x, data_m[:, col], color='tab:red', label='raw',ls=':')
                plt.plot(x, imputed_col, color='tab:blue', label='imputed', ls='-')
                plt.show()