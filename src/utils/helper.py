from torch import nn
import numpy as np
import torch as t


def init_params(m, mean = 0.0, std = 0.01):

    """
    Normal Weight and Zero Bias Initializer
    """
    
    nn.init.normal_(m.weight, mean, std)
    m.bias.data.zero_()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor
    return tensor    

def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()

