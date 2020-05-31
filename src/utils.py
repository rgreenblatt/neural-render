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


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots_vals'))):
    def __call__(self, t):
        knots = [x[0] for x in self.knots_vals]
        vals = [x[1] for x in self.knots_vals]
        return np.interp([t], knots, vals)[0]


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
        linear = torch.linspace(0.0, 1.0,
                                steps=width).type(dtype).to(device).view(
                                    width, 1)
        x_ch = linear.expand(width, width)
        y_ch = linear.view(1, width).expand(width, width)
        out[width] = torch.cat(
            (x_ch.view(1, width, width), y_ch.view(1, width, width)),
            dim=0).view(1, 2, width, width)

    return out


# Note: expects tensor type in standard format (N x C x H x W)
def linear_to_srgb(img):
    return torch.clamp(torch.where(img <= 0.0031308, 12.92 * img,
                                   1.055 * torch.pow(img, 1 / 2.4) - 0.055),
                       min=0.0,
                       max=1.0)


# Note: expects numpy format image (H x W x C)
# expects square format and clean multiple
def resize(img, output_size):
    input_size = img.shape[0]
    bin_size = input_size // output_size

    assert bin_size * output_size == input_size, "multiple must be exact"

    return img.reshape(
        (output_size, bin_size, output_size, bin_size, 3)).mean(3).mean(1)

class LossTracker():
    def __init__(self, reduce_tensor):
        super().__init__()

        self.reduce_tensor = reduce_tensor

        self.total_loss = None
        self.loss_steps = 0

    def update(self, loss):
        # clone probably not needed
        if self.total_loss is None:
            self.total_loss = loss.detach().clone()
        else:
            self.total_loss += loss.detach().clone()
        self.loss_steps += 1

    def query_reset(self):
        if self.total_loss is None:
            return None

        avg_loss = self.reduce_tensor(self.total_loss).item() / self.loss_steps
        self.total_loss = None
        self.loss_steps = 0

        return avg_loss


class ImageTracker():
    def __init__(self):
        super().__init__()

        self.images = None

    def update(self, images):
        # clone probably not needed
        images = linear_to_srgb(images.detach().clone().float())

        if self.images is None:
            self.images = images
        else:
            self.images = torch.cat((self.images, images), dim=0)

    def query_reset(self):
        if self.images is None:
            return None

        cpu_images = self.images.cpu()
        self.images = None

        return cpu_images
