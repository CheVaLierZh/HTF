import random

import numpy as np


class MissingGenerator(object):

    def __init__(self, missing_pattern='one', seed=100):
        """

        Args:
            missing_pattern: pattern of missing blocks:
                                'one': only one variate time series missing
                                'overlap': multi-time series missing, and blocks overlap, but no all time - series
                                            missing at same time
                                'blackout': multi-time series missing, and contains all blocks overlap, means missing at
                                             same time
                                'random': MCAR missing pattern
                                default: 'one'
            seed:
        """
        self.pattern = missing_pattern
        self.seed = seed

    def set_missing_percent(self, missing_percent=0.4):
        """

        :param missing_percent: missing percent of time points of total in pattern "one"
        :return:
        """
        # assert self.pattern == 'one'
        self.missing_percent = missing_percent

    def set_missing_number(self, missing_number, missing_percent=0.2):
        """
        In pattern "overlap"

        :param missing_number: number of missing variates in data
        :param missing_percent: missing percent of time points of total in each variate
        :return:
        """
        # assert self.pattern == 'overlap'
        self.missing_number = missing_number
        self.missing_percent = missing_percent

    def set_missing_size(self, missing_size=200):
        """

        :param missing_size: missing number of time points in data in pattern "blackout"
        :return:
        """
        # assert self.pattern == 'blackout'
        self.missing_size = missing_size

    def set_missing_rate(self, missing_rate=0.4):
        """

        :param missing_rate: missing prob of each time points in pattern "random"
        :return:
        """
        # assert self.pattern == 'random'
        self.missing_rate = missing_rate

    def generate(self, data):
        """

        :param data: the data to generate missing mask
        :return: the mask indicate which time points in data is missing, and is marked as 0, otherwise 1
        """
        random.seed(self.seed)
        mask = np.ones(data.shape)

        if self.pattern == 'disjoint':
            missing_variates_number = int(data.shape[0] * self.missing_number)

            missing_block_size = self.missing_size
            tmp = np.arange(0, data.shape[0])
            np.random.shuffle(tmp)
            missing_variates = np.sort(tmp[:missing_variates_number])

            pos = np.random.randint(data.shape[1] // 5, data.shape[1] // 4)
            for variable in missing_variates:
                if pos + missing_block_size < mask.shape[1]:
                    mask[variable, pos:pos+missing_block_size] = 0
                    pos = pos + missing_block_size + np.random.randint(5, 20)


        if self.pattern == 'overlap':
            missing_variates_number = int(data.shape[0] * self.missing_number)
            # print(missing_variates_number)
            missing_block_size = self.missing_size
            tmp = np.arange(0, data.shape[0])
            np.random.shuffle(tmp)
            missing_variates = np.sort(tmp[:missing_variates_number])

            pos = np.random.randint(data.shape[1] // 5, data.shape[1] // 2)
            for variable in missing_variates:
                if pos + missing_block_size < mask.shape[1]:
                    mask[variable, pos:pos + missing_block_size] = 0
                    pos = pos + missing_block_size // 2

        if self.pattern == 'blackout':
            non_overlap = 0
            pos = np.random.randint(int((data.shape[1] - non_overlap) / 4), int((data.shape[1] - non_overlap) / 5 * 3))
            for variable in np.arange(data.shape[0]):
                mask[variable, pos:pos + self.missing_size] = 0
                pos = pos + non_overlap

        return mask


if __name__ == '__main__':
    generator_ = MissingGenerator(missing_pattern='blackout')
    generator_.set_missing_size(missing_size=40)
    data_ = np.random.randn(5, 100)

    mask_ = generator_.generate(data_)
    print(mask_)


