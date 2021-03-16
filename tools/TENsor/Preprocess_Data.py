import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Algorithm.tools.utils import _find_max_continuous_missing
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class PieceDataPreprocessor(object):
    def __init__(self, seq_len, batch_size, scale_method=None):
        super(PieceDataPreprocessor, self).__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size

        if scale_method is None:
            self.scaler = None
        elif scale_method == 'MinMax':
            self.scaler = MinMaxScaler()
        else:
            raise Exception('not found scaling method %s' % scale_method)

    def inverse_scale(self, X):
        return self.scaler.inverse_transform(X)

    def dpreprocess(self, data, mask):
        data_norm = data
        if self.scaler is not None:
            data_norm = self.scaler.fit_transform(X=data)

        n_timepoints, n_features = data_norm.shape

        num_max_blank = _find_max_continuous_missing(mask, axis=0)
        # print(num_max_blank)
        if num_max_blank > self.seq_len - 2:
            old_seq_len = self.seq_len
            self.seq_len = max(num_max_blank + 7, int(num_max_blank * 2))
            print('Invalid seq_len parameter %d, set to default max_blank * 2, %d' % (old_seq_len, self.seq_len))

        x_origin_tol = []
        x_tol = []
        m_tol = []
        t_tol = []

        for i in range(n_timepoints - self.seq_len + 1):
            x_origin = data_norm[i:i+self.seq_len, :]
            x_origin_tol.append(torch.Tensor(x_origin))
            m = mask[i:i+self.seq_len, :]
            m_tol.append(torch.Tensor(m))

            x = x_origin.copy()
            x[m == 0] = 0.0
            x_tol.append(torch.Tensor(x))

            t = np.ones((self.seq_len, n_features))
            for k in range(n_features):
                for j in range(1, self.seq_len, 1):
                    if m[j, k] == 0:
                        t[j, k] = t[j-1, k] + 1.
            t_tol.append(torch.Tensor(t))

        r = torch.randperm(len(x_tol))
        x_origin_tol = torch.stack(x_origin_tol)
        x_tol = torch.stack(x_tol)
        m_tol = torch.stack(m_tol)
        t_tol = torch.stack(t_tol)

        x_origin_tol = x_origin_tol[r]
        x_tol = x_tol[r]
        m_tol = m_tol[r]
        t_tol = t_tol[r]

        print('Generate train dataset with size ', x_tol.size())
        ds = TensorDataset(x_tol, m_tol, t_tol, x_origin_tol)
        train_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        return train_dl, r, data_norm, mask

    def readfile(self, datafile, maskfile, delim, row_is_time=True, skip_row=1):
        data = np.loadtxt(datafile, delimiter=delim, skiprows=skip_row)
        mask = np.loadtxt(maskfile, delimiter=delim, skiprows=skip_row)

        if not row_is_time:
            data = data.T
            mask = mask.T

        data_norm = data
        if self.scaler is not None:
            data_norm = self.scaler.fit_transform(X=data)

        return data_norm, mask


    def preprocess(self, datafile, maskfile, delim, row_is_time=True, skip_row=1):
        """

        :param datafile:
        :param maskfile:
        :param delim:
        :param row_is_time:
        :param skip_row:
        :return:
        """
        data = np.loadtxt(datafile, delimiter=delim, skiprows=skip_row)
        mask = np.loadtxt(maskfile, delimiter=delim, skiprows=skip_row)

        if not row_is_time:
            data = data.T
            mask = mask.T

        data_norm = data
        if self.scaler is not None:
            data_norm = self.scaler.fit_transform(X=data)

        n_timepoints, n_features = data_norm.shape

        num_max_blank = _find_max_continuous_missing(mask, axis=0)
        # print(num_max_blank)
        if num_max_blank > self.seq_len - 2:
            old_seq_len = self.seq_len
            self.seq_len = max(num_max_blank + 7, int(num_max_blank * 2))
            print('Invalid seq_len parameter %d, set to default max_blank * 2, %d' % (old_seq_len, self.seq_len))

        x_origin_tol = []
        x_tol = []
        m_tol = []
        t_tol = []

        for i in range(n_timepoints - self.seq_len + 1):
            x_origin = data_norm[i:i+self.seq_len, :]
            x_origin_tol.append(torch.Tensor(x_origin))
            m = mask[i:i+self.seq_len, :]
            m_tol.append(torch.Tensor(m))

            x = x_origin.copy()
            x[m == 0] = 0.0
            x_tol.append(torch.Tensor(x))

            t = np.ones((self.seq_len, n_features))
            for k in range(n_features):
                for j in range(1, self.seq_len, 1):
                    if m[j, k] == 0:
                        t[j, k] = t[j-1, k] + 1.
            t_tol.append(torch.Tensor(t))

        r = torch.randperm(len(x_tol))
        x_origin_tol = torch.stack(x_origin_tol)
        x_tol = torch.stack(x_tol)
        m_tol = torch.stack(m_tol)
        t_tol = torch.stack(t_tol)

        x_origin_tol = x_origin_tol[r]
        x_tol = x_tol[r]
        m_tol = m_tol[r]
        t_tol = t_tol[r]

        print('Generate train dataset with size ', x_tol.size())
        ds = TensorDataset(x_tol, m_tol, t_tol, x_origin_tol)
        train_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        return train_dl, r, data_norm, mask

    @staticmethod
    def get_total_from_dl(train_dl):
        x_tol = []
        m_tol = []
        t_tol = []
        x_origin_tol = []
        for x, m, t, output_train in train_dl:
            x_tol.append(x)
            m_tol.append(m)
            t_tol.append(t)
            x_origin_tol.append(output_train)
        return x_tol, m_tol, t_tol, x_origin_tol

    @staticmethod
    def DatafromPieceToComplete(x, ind_order):
        y = x[0]
        for i in range(1, len(x), 1):
            y = torch.cat((y, x[i]), dim=0)

        num_piece, seq_len, n_feats = y.size()
        n_timepoints = seq_len + num_piece - 1

        data = np.zeros((n_timepoints, n_feats))
        cnt = np.zeros((n_timepoints, n_feats))

        for i in range(ind_order.size()[0]):
            data[ind_order[i]:ind_order[i]+seq_len, :] += y[i].numpy()
            cnt[ind_order[i]:ind_order[i]+seq_len, :] += 1

        for i in range(n_timepoints):
            data[i, :] /= cnt[i, :]

        return data






