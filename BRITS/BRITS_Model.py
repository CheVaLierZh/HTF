import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter
from Algorithm.tools.TENsor.utils import reverse_tensor
from torch.autograd import Variable


class FeatureRegression(nn.Module):
    def __init__(self, input_dim):
        super(FeatureRegression, self).__init__()
        self.W = Parameter(torch.Tensor(input_dim, input_dim))
        self.b = Parameter(torch.Tensor(input_dim))

        m = torch.ones(input_dim, input_dim) - torch.eye(input_dim, input_dim)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_dim, output_dim, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.W = Parameter(torch.Tensor(output_dim, input_dim))
        self.b = Parameter(torch.Tensor(output_dim))

        if self.diag:
            assert(input_dim == output_dim)
            m = torch.eye(input_dim, input_dim)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.diag:
            gamma = F.relu(F.linear(x, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(x, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(RITS, self).__init__()
        self.hid_dim = hid_dim

        self.rnn_cell = nn.LSTMCell(input_size=input_dim * 2, hidden_size=hid_dim)

        self.temp_decay_h = TemporalDecay(input_dim, hid_dim, diag=False)
        self.temp_decay_m = TemporalDecay(input_dim, input_dim, diag=True)

        self.time_reg = nn.Linear(hid_dim, input_dim)
        self.feat_reg = FeatureRegression(input_dim)

        self.weight_combine = nn.Linear(input_dim * 2, input_dim)

        self.dropout = nn.Dropout(p=.25)
        # self.out = nn.Linear(hid_dim, 1)   since we donot need predict labels

    def forward(self, x, m, t):
        """

        :param x: tensor(batch_size, seq_len, n_feats)
        :param m:
        :param t:
        :return:
        """
        batch_size, seq_len, n_feats = x.size()

        h = Parameter(torch.zeros((batch_size, self.hid_dim)))
        c = Parameter(torch.zeros((batch_size, self.hid_dim)))
        h = h.to(x.device.type)
        c = c.to(x.device.type)

        loss = 0.0

        imputations = []

        for l in range(seq_len):
            xl = x[:, l, :]
            ml = m[:, l, :]
            tl = t[:, l, :]

            gamma_h = self.temp_decay_h(tl)
            gamma_m = self.temp_decay_m(tl)

            # print(h.device)
            # print(gamma_h.device)
            h = h * gamma_h
            xl_hat = self.time_reg(h)

            loss += torch.sum(torch.pow(xl-xl_hat, 2) * ml) / (torch.sum(ml) + 1e-5)

            xl_c = ml * xl + (1-ml) * xl_hat

            zl_hat = self.feat_reg(xl_c)
            loss += torch.sum(torch.pow(xl - zl_hat, 2) * ml) / (torch.sum(ml) + 1e-5)

            beta = self.weight_combine(torch.cat([gamma_m, ml], dim=1))

            cl_hat = beta * zl_hat + (1 - beta) * xl_hat
            loss += torch.sum(torch.pow(xl - cl_hat, 2) * ml) / (torch.sum(ml) + 1e-5)

            cl_c = ml * xl + (1 - ml) * cl_hat
            # cl_c = cl_hat
            imputations.append(cl_c)

            inputs_l = torch.cat([cl_c, ml], dim=1)

            h, c = self.rnn_cell(inputs_l, (h, c))

        imputations = torch.stack(imputations).permute(1, 0, 2)

        return imputations, loss / seq_len


class BRITS(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(BRITS, self).__init__()
        self.rits_f = RITS(input_dim, hid_dim)
        self.rits_b = RITS(input_dim, hid_dim)

    def forward(self, x, m, t):
        imputations_f, loss_f = self.rits_f(x, m, t)

        x_reverse = reverse_tensor(x)
        m_reverse = reverse_tensor(m)
        t_reverse = reverse_tensor(t)
        imputations_b, loss_b = self.rits_b(x_reverse, m_reverse, t_reverse)

        imputations_b = reverse_tensor(imputations_b)
        imputations = (imputations_f + imputations_b) / 2

        consistency_loss = torch.pow(imputations_f - imputations_b, 2).mean()

        loss = loss_f + loss_b + consistency_loss

        return imputations, loss

