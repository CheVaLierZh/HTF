import numpy as np


def cut_data(file, preserve_points=1000):
    data = np.loadtxt(file, delimiter=' ', dtype=np.float32)
    if data.shape[0] > data.shape[1]:
        data = data.T   #  transpose (n_timepoints, n_features) format
    data = data[:, :preserve_points]
    np.savetxt(file, data, fmt='%f', delimiter=' ')


if __name__ == '__main__':
    file = '../../Dataset/Electricity/electricity_normal.txt'
    cut_data(file, preserve_points=1000)