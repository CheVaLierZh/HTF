import torch
import torch.optim as optim
import torch.nn as nn
from Algorithm.BRITS.BRITS_Model import BRITS
from Algorithm.tools.performance import fn_timer
from Algorithm.base import BaseRestorer
from Algorithm.tools.TENsor.Preprocess_Data import PieceDataPreprocessor
from Algorithm.tools.performance import memory_usage
import time


SEED = 1024
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BRITSLearner(BaseRestorer):
    def __init__(self, optim_method, max_epoch, lr):
        self.mem_start =memory_usage()
        super(BRITSLearner, self).__init__()
        self.device = device

        self.brits = None
        self.optim_method = optim_method
        self.lr = lr

        self.max_epoch = max_epoch

        self.dp = None

        self.hid_dim = 0

        self.time_cost = 0
        self.real_iters = 0

    def get_optimizer(self, params, lr):
        optimizer = None
        if self.optim_method == 'Adam':
            optimizer = optim.Adam(params, lr)
        elif self.optim_method == 'SGD':
            optimizer = optim.SGD(params, lr)
        else:
            raise Exception("not found optim method called %s" % self.optim_method)
        return optimizer

    # @fn_timer
    def train(self, train_dl, hid_dim):
        """

        :param train_dl:
        :return:
        """
        batch_size, seq_len, n_feats = (next(iter(train_dl))[0]).size()

        self.brits = BRITS(n_feats, hid_dim)
        self.brits.to(self.device)

        optimizer = self.get_optimizer(self.brits.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50, 90], gamma=0.1)

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            n_iters = 0
            t_start = time.time()
            for x, m, t, output_train in train_dl:
                x = x.to(self.device)
                m = m.to(self.device)
                t = t.to(self.device)
                output_train = output_train.to(self.device)

                optimizer.zero_grad()
                imputations, loss = self.brits(x, m, t)

                loss.backward()
                optimizer.step()

                running_loss += loss.to("cpu").item()
                n_iters += 1

            scheduler.step()
            print('[%d] loss: %f' % (epoch + 1, running_loss / n_iters))
            t_end = time.time()
            print('consuming time: %.2f s' % (t_end - t_start))
            self.time_cost += t_end - t_start
            self.real_iters = epoch+1

    def impute(self, dl):
        imputations = []
        for x, m, t, output_train in dl:
            x = x.to(self.device)
            m = m.to(self.device)
            t = t.to(self.device)

            with torch.no_grad():
                imputation, loss = self.brits(x, m, t)

            imputations.append(imputation.to("cpu"))

        return imputations

    def accept_data_preprocessor(self, dp):
        self.dp = dp

    def set_hidden_dim(self, hidden_dim):
        self.hid_dim = hidden_dim

    def restore(self, data, mask):
        train_dl, ind_order, data_m, mask_m = self.dp.dpreprocess(data, mask)

        self.train(train_dl, hid_dim=self.hid_dim)
        imputed = self.impute(train_dl)

        _, mask, __, origin = self.dp.get_total_from_dl(train_dl)

        imputed = self.dp.DatafromPieceToComplete(imputed, ind_order=ind_order)

        mem_cur = memory_usage()
        self.avg_time_cost = self.time_cost / self.real_iters
        self.mem_cost = mem_cur - self.mem_start
        print('Average time every iteration: %.2f s' % (self.time_cost / self.real_iters))
        print('Consuming memory: %.2f MB' % (mem_cur - self.mem_start))
        return imputed, data_m, mask_m

    def get_time_space(self):
        return self.avg_time_cost, self.mem_cost

