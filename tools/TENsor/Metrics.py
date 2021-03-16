import torch


def rmse(x, m, x_impute):
    m = 1-m
    return (torch.sum(((x - x_impute) * m) ** 2) / torch.sum(m)).item()
