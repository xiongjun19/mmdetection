# coding=utf8

import torch
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS

class Mish(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Mish, self).__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


ACTIVATION_LAYERS.register_module(module=Mish)
