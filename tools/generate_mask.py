from Algorithm.tools.missing_generator import MissingGenerator
import numpy as np
import os


if __name__ == '__main__':
    # generator = MissingGenerator(missing_pattern='disjoint')
    # generator.set_missing_number(missing_number=0.4)
    # generator.set_missing_size(missing_size=100)
    generator = MissingGenerator(missing_pattern='overlap')
    generator.set_missing_number(missing_number=0.4)
    generator.set_missing_size(missing_size=100)
    # generator = MissingGenerator(missing_pattern='blackout')
    # generator.set_missing_size(missing_size=100)

    n_times = 5

    dataset_path = '..\\..\\Dataset'
    data_dir = 'Temperature'
    data_name = 'temp_normal'
    data_file_suffix = '.txt'
    data_file = data_name + data_file_suffix

    path = os.path.dirname(os.path.abspath(__file__))
    r_dir = os.path.join(os.path.join(path, dataset_path), data_dir)
    r_file = os.path.join(r_dir, data_file)
    data = np.loadtxt(r_file, delimiter=' ', dtype=np.float32)
    if data.shape[0] > data.shape[1]:
        data = data.T
        np.savetxt(os.path.join(path, r_file), data, fmt='%f', delimiter=' ')

    for i in range(n_times):
        mask = generator.generate(data)

        mask_file = data_name + '_' + generator.pattern + str(i+1) + data_file_suffix
        np.savetxt(os.path.join(r_dir, mask_file), mask, fmt='%d', delimiter=' ')