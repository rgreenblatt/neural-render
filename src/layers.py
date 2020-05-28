import functools
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter as P
import numpy as np

from utils import (Swish, MemoryEfficientSwish)
from pydbg import dbg


# Simple function to handle groupnorm norm stylization
def get_groupnorm_groups(num_ch, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(num_ch // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16

    return groups


# From big gan pytorch
class configurable_norm(nn.Module):
    def __init__(
        self,
        output_size,
        input_gain_bias,
        eps=1e-5,
        momentum=0.1,
        norm_style='bn',
    ):
        super().__init__()

        self.output_size = output_size
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        self.input_gain_bias = input_gain_bias

        if not self.input_gain_bias:
            self.gain = nn.Parameter(torch.Tensor(output_size))
            self.bias = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

        if self.norm_style == 'bn':
            self.norm = nn.BatchNorm2d(output_size,
                                       eps=eps,
                                       momentum=momentum,
                                       affine=False)
        elif self.norm_style == 'in':
            self.norm = nn.InstanceNorm2d(output_size,
                                          eps=eps,
                                          momentum=momentum,
                                          affine=False)
        elif self.norm_style.startswith('gn'):
            self.norm = nn.GroupNorm(get_groupnorm_groups(
                output_size, self.norm_style),
                                     output_size,
                                     eps=eps,
                                     affine=False)
        elif self.norm_style == 'nonorm':
            self.norm = lambda x: x

    def reset_parameters(self):
        if not self.input_gain_bias:
            nn.init.ones_(self.gain)
            nn.init.zeros_(self.bias)

    def forward(self, x, gain=None, bias=None):
        out = self.norm(x)

        if self.input_gain_bias:
            gain = gain.view(gain.size(0), -1, 1, 1)
            bias = bias.view(bias.size(0), -1, 1, 1)
        else:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)

        return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size},'
        s += ' norm_style={norm_style}'
        return s.format(**self.__dict__)


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))

    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)

    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self,
                 input_size,
                 key_size,
                 output_size,
                 n_heads,
                 query_is_input=False):
        super().__init__()

        self.query_is_input = query_is_input

        self._proj_k = nn.Linear(input_size, key_size)
        self._proj_v = nn.Linear(input_size, output_size)

        if not self.query_is_input:
            self._proj_q = nn.Linear(input_size, key_size)

            # added here (TODO: ablation etc important)
            self._proj_c = nn.Linear(input_size, output_size)

        self.n_heads = n_heads

    def forward(self, x, input_query=None):
        """
        x, q(query), k(key), v(value), c(carry) : (B(batch), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W

        For the key and query, W correponds to key_size (WK).
        For the value, W correponds to output_size (WO).
        """

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W)

        if self.query_is_input:
            q = input_query
        else:
            q = self._proj_q(x)

            # I think this might not be needed in general
            c = self._proj_c(x)

        k = self._proj_k(x)
        v = self._proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)) for x in [q, k, v])

        # (B, S, H, W) -transpose-> (B, H, S, W)
        # if using input_query, then b_q is:
        # (B, S*, H, W) -transpose-> (B, H, S*, W)
        # S* can be anything (output sequence size)

        q, k, v = (x.transpose(1, 2) for x in [q, k, v])

        # (B, H, S, WK) @ (B, H, WK, S) -> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))

        # (B, H, S, S) -softmax-> (B, H, S, S) (could have dropout)
        scores = F.softmax(scores, dim=-1)

        # (B, H, S, S) @ (B, H, S, WV) -> (B, H, S, WV) -trans-> (B, S, H, WV)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        out = merge_last(h, 2)


        if not self.query_is_input:
            out = out + c

        return out


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()

        self._fc1 = nn.Linear(cfg.size, cfg.hidden_ff_size)
        self._fc2 = nn.Linear(cfg.hidden_ff_size, cfg.size)
        self._swish = MemoryEfficientSwish()

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self._fc2(self._swish(self._fc1(x)))

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard
           (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of
            swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, size, variance_epsilon=1e-12):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


TransformerCfg = collections.namedtuple(
    'TransformerCfg', ['size', 'n_heads', 'n_layers', 'hidden_ff_size'])

class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks and parameter sharing"""
    def __init__(self, cfg):
        super().__init__()

        self.n_layers = cfg.n_layers
        self._attn = MultiHeadedSelfAttention(cfg.size, cfg.size, cfg.size,
                                              cfg.n_heads)
        self._proj = nn.Linear(cfg.size, cfg.size)
        self._norm1 = LayerNorm(cfg.size)
        self._pwff = PositionWiseFeedForward(cfg)
        self._norm2 = LayerNorm(cfg.size)

    def forward(self, h):
        # NOTE to self, masking may be useful...
        for _ in range(self.n_layers):
            prev_layer = h
            h = self._attn(h)
            h = self._norm1(prev_layer + self._proj(h))
            h = self._norm2(h + self._pwff(h))

        return h

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard
           (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of
            swish.
        """

        self._pwff.set_swish(memory_efficient)

SeqToImageStartCfg = collections.namedtuple(
    'SeqToImageStartCfg',
    ['start_ch', 'ch_per_head', 'start_width', 'seq_size'])


class SeqToImageStart(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.ch_groups = self.cfg.start_ch // self.cfg.ch_per_head
        n_heads = self.ch_groups * self.cfg.start_width**2
        output_size = self.cfg.start_width**2 * self.cfg.start_ch

        self._avg_to_key = nn.Linear(self.cfg.seq_size, output_size)

        # I think attn doesn't have a bias, so we use it here
        self._count_to_output = nn.Linear(1, output_size, bias=True)

        self._attn = MultiHeadedSelfAttention(self.cfg.seq_size,
                                              output_size,
                                              output_size,
                                              n_heads,
                                              query_is_input=True)

    def forward(self, x, y=None):
        # Uses average and count to produce reduce_key
        avgs = x.mean(1, keepdim=True)

        key = self._avg_to_key(avgs)
        attention_output = self._attn(x, key)
        count_v = self._count_to_output(
            torch.tensor([x.size(1)], dtype=x.dtype,
                         device=x.device)).view(1, 1, -1)

        output = attention_output + count_v

        # return as NxCxHxW
        return output.reshape(output.size(0), self.cfg.start_ch,
                              self.cfg.start_width, self.cfg.start_width)


MBConvCfg = collections.namedtuple('MBConvCfg', [
    'input_ch', 'output_ch', 'upsample', 'expand_ratio', 'kernel_size',
    'norm_style', 'se_ratio', 'show_position'
])


class MBConvGBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block. (with upsampling and class bn)

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # pytorch's difference from tensorflow
        self.has_se = (self.cfg.se_ratio is
                       not None) and (0 < self.cfg.se_ratio <= 1)

        # Expansion phase (Inverted Bottleneck)
        inp = self.cfg.input_ch  # number of input channels
        # number of output channels
        oup = self.cfg.input_ch * self.cfg.expand_ratio

        self.which_norm = functools.partial(configurable_norm,
                                            input_gain_bias=False,
                                            norm_style=cfg.norm_style)

        self._bn0 = self.which_norm(inp)

        if self.cfg.show_position:
            inp += 2

        # if expand ratio == 1 this probably isn't needed...
        self._expand_conv = nn.Conv2d(in_channels=inp,
                                      out_channels=oup,
                                      kernel_size=1,
                                      bias=False)
        self._bn1 = self.which_norm(oup, input_gain_bias=False)

        # Depthwise convolution phase
        k = self.cfg.kernel_size
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            bias=False,
            padding=k // 2)
        self._bn2 = self.which_norm(oup, input_gain_bias=False)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self.cfg.input_ch * self.cfg.se_ratio))
            # TODO: this may not be efficient - use dense...
            # (probably doesn't matter)
            self._se_reduce = nn.Conv2d(in_channels=oup,
                                        out_channels=num_squeezed_channels,
                                        kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels,
                                        out_channels=oup,
                                        kernel_size=1)

        # Pointwise convolution phase
        final_oup = self.cfg.output_ch
        self._project_conv = nn.Conv2d(in_channels=oup,
                                       out_channels=final_oup,
                                       kernel_size=1,
                                       bias=False)
        self._swish = MemoryEfficientSwish()

        self._upsample = functools.partial(F.interpolate, scale_factor=2)

    def forward(self, inputs, position_ch):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs

        x = self._bn0(x)
        x = self._swish(x)

        if self.cfg.show_position:
            expanded_pos = position_ch[x.size(2)].expand(x.size(0), -1, -1, -1)

            x = torch.cat((x, expanded_pos), dim=1)

        x = self._expand_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.cfg.upsample:
            x = self._upsample(x)
            inputs = self._upsample(inputs)

        inputs = inputs[:, :self.cfg.output_ch]

        x = self._depthwise_conv(x)
        x = self._bn2(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)

        return torch.cat(
            (x[:, :self.cfg.input_ch] + inputs, x[:, self.cfg.input_ch:]),
            dim=1)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


ImageToSeqCfg = collections.namedtuple(
    'ImageToSeqCfg', ['image_ch', 'seq_size', 'n_heads'])


class ImageToSeq(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # roughly, this layer is 2 element multi headed attention
        self._pool = nn.AdaptiveMaxPool2d(1)
        self._proj_k = nn.Linear(self.cfg.image_ch, self.cfg.seq_size)
        self._proj_v = nn.Linear(self.cfg.image_ch, self.cfg.seq_size)
        self._proj_q = nn.Linear(self.cfg.seq_size, self.cfg.seq_size)

    # x is seq (BS x D), y is image type data (B x C x H x W)
    def forward(self, x, y):
        # (B x C x H x W) -pool-> (B x C)
        pooled = self._pool(y).view(y.size(0), -1)

        # (B x C) -proj-> (B x D) -unsqueeze-> (B x 1 x D)
        k = self._proj_k(pooled)[:, None]
        v = self._proj_v(pooled)[:, None]

        # (B x S x D) -proj-> (B x S x D)
        q = self._proj_q(x)

        # (B x S x D) -split-> (B x S x H x W)
        x, q, k, v = (split_last(x, (self.cfg.n_heads, -1))
                      for x in [x, q, k, v])

        # (B, S, H, W) -transpose-> (B, H, S, W)
        x, q, k, v = (x.transpose(1, 2) for x in [x, q, k, v])


        # (B, H, S, WK) @ (B, H, WK, 1) -> (B, H, S, 1)
        scores = (q @ k.transpose(-2, -1))
        scores = torch.sigmoid(scores)

        out = x * (1 - scores) + v * scores

        # (B, H, S, WV) -trans-> (B, S, H, WV)
        out = out.transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        out = merge_last(out, 2)

        return out


SeqToImageCfg = collections.namedtuple(
    'SeqToImageCfg', ['image_ch', 'seq_size', 'output_ch', 'n_heads'])


class SeqToImage(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._proj_k = nn.Conv2d(self.cfg.image_ch,
                                 self.cfg.output_ch,
                                 kernel_size=1)
        self._attn = MultiHeadedSelfAttention(self.cfg.seq_size,
                                              self.cfg.output_ch,
                                              self.cfg.output_ch,
                                              self.cfg.n_heads,
                                              query_is_input=True)

    # x is seq (B x S x D), y is image type data (B x C x H x W)
    def forward(self, x, y):
        k = self._proj_k(y)

        batch_size = y.size(0)
        height = y.size(2)
        width = y.size(3)
        image_size = height * width

        # to NHWC
        k = k.permute(0, 2, 3, 1)

        k = k.view(batch_size, image_size, -1)

        # attention is applied elementwise to "pixels" (similar to global
        # attention from AttnNet - just implementation differences)
        added_values = self._attn(x, k)
        added_values = added_values.permute(0, 2, 1)
        added_values = added_values.view(batch_size, -1, height, width)

        return torch.cat((y, added_values), dim=1)


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, name='attention'):
        super(Attention, self).__init__()

        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch,
                               self.ch // 8,
                               kernel_size=1,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv2d(self.ch,
                             self.ch // 8,
                             kernel_size=1,
                             padding=0,
                             bias=False)
        self.g = nn.Conv2d(self.ch,
                           self.ch // 2,
                           kernel_size=1,
                           padding=0,
                           bias=False)
        self.o = nn.Conv2d(self.ch // 2,
                           self.ch,
                           kernel_size=1,
                           padding=0,
                           bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None, position_ch=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2,
                                                    x.shape[2], x.shape[3]))
        return self.gamma * o + x
