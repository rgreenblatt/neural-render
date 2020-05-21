"""utils.py - Helper functions for building the model
   source: https://github.com/lukemelas/EfficientNet-PyTorch
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

from pathlib import Path
import math
from functools import partial
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


# Swish and MemoryEfficientSwish: Two implementations of the method
# round_filters and round_repeats:
#     Functions to calculate params for scaling model width and depth ! ! !
# get_width_and_height_from_size and calculate_output_image_size
# drop_connect: A structural design
# get_same_padding_conv2d:
#     Conv2dDynamicSamePadding
#     Conv2dStaticSamePadding
# get_same_padding_maxPool2d:
#     MaxPool2dDynamicSamePadding
#     MaxPool2dStaticSamePadding
#     It's an additional function, not used in EfficientNet,
#     but can be used in other model (such as EfficientDet).
# Identity: An implementation of identical mapping


# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def get_position_ch(min_width, max_width, dtype, device):
    out = {}
    for i in range(int(math.log2(min_width)), int(math.log2(max_width)) + 1):
        width = 2**i
        linear = torch.linspace(0.0,
                                1.0,
                                steps=width,
                                dtype=dtype,
                                device=device).view(width, 1)
        x_ch = linear.expand(width, width)
        y_ch = linear.view(1, width).expand(width, width)
        out[width] = torch.cat(
            (x_ch.view(1, width, width), y_ch.view(1, width, width)),
            dim=0).view(1, 2, width, width)

    return out
