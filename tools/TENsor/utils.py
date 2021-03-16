import torch
import numpy as np


def reverse_tensor(x):
    device = x.device.type
    x = x.to("cpu")
    ret = torch.from_numpy(np.flip(x.detach().numpy(), 1).copy())
    return ret.to(device)

