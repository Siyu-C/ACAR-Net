import torch.nn as nn

from .basic import *


def model_entry(config):
    return globals()[config['type']](**config['kwargs'])


class AVA_neck(nn.Module):
    def __init__(self, config):
        super(AVA_neck, self).__init__()
        self.module = model_entry(config)

    def forward(self, data):
        return self.module(data)
