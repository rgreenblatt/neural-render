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
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


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

        self.track_running_stats = self.norm_style in ['bn', 'in']

        if self.track_running_stats:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.stored_mean.zero_()
            self.stored_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if not self.input_gain_bias:
            nn.init.ones_(self.gain)
            nn.init.zeros_(self.bias)

    def forward(self, x, gain=None, bias=None):
        if self.norm_style == 'bn':
            out = F.batch_norm(x, self.stored_mean, self.stored_var, None,
                               None, self.training, self.momentum, self.eps)
        elif self.norm_style == 'in':
            out = F.instance_norm(x, self.stored_mean, self.stored_var, None,
                                  None, self.training, self.momentum, self.eps)
        elif self.norm_style.startswith('gn'):
            out = groupnorm(x, self.norm_style)
        elif self.norm_style == 'nonorm':
            out = x

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

    def forward(self, x, splits, input_query=None):
        """
        We incorporate sequences length into the batch dimension and
        include "splits" as an input

        TODO: make this efficient with respect to long sequences

        x, q(query), k(key), v(value), c(carry) : (BS(batch * seq), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W

        For the key and query, W correponds to key_size (WK).
        For the value, W correponds to output_size (WO).
        """

        # (BS, D) -proj-> (BS, D) -split-> (BS, H, W)

        if self.query_is_input:
            q = input_query
        else:
            q = self._proj_q(x)

            # I think this might not be needed in general
            c = self._proj_c(x)

        k = self._proj_k(x)
        v = self._proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)) for x in [q, k, v])

        h_values = []

        # handle each batch element independently (for now at least)
        for i, (prev, after) in enumerate(splits):
            # (BS, H, W) -slice-> (S, H, W) -transpose-> (H, S, W)
            # -unsqueeze-> (1, H, S, W)
            # we mostly unsqueeze for consistancy with standard models
            # if using input_query, then b_q is:
            # (B, S*, H, W) -index-> (S*, H, W) -transpose-> (H, S*, W)
            # -unsqueeze-> (1, H, S*, W)
            # S* can be anything

            if self.query_is_input:
                b_q = q[i]
            else:
                b_q = q[prev:after]

            b_k, b_v = (x[prev:after] for x in [k, v])
            b_q, b_k, b_v = (x.transpose(0, 1)[None] for x in [b_q, b_k, b_v])

            # (1, H, S, WK) @ (1, H, WK, S) -> (1, H, S, S)
            scores = b_q @ b_k.transpose(-2, -1) / np.sqrt(b_k.size(-1))

            # (1, H, S, S) -softmax-> (1, H, S, S) (could have dropout)
            scores = F.softmax(scores, dim=-1)

            # (1, H, S, S) @ (1, H, S, WV) -> (1, H, S, WV)
            # -trans-> (1, S, H, WV)
            h = (scores @ b_v).transpose(1, 2).contiguous()
            # -merge-> (S, D)
            h = merge_last(h, 2).squeeze(0)

            h_values.append(h)

        if self.query_is_input:
            out = torch.stack(h_values)
        else:
            out = torch.cat(h_values, dim=0)

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

    def forward(self, h, splits):
        # NOTE to self, masking may be useful...
        for _ in range(self.n_layers):
            prev_layer = h
            h = self._attn(h, splits)
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
        self._attn = MultiHeadedSelfAttention(self.cfg.seq_size,
                                              output_size,
                                              output_size,
                                              n_heads,
                                              query_is_input=True)

    def forward(self, x, splits, y=None):
        # Uses average as reduce_key and produces NxCxHxW tensor
        avgs = []
        for (prev, after) in splits:
            count = after - prev
            avgs.append(x[prev:after].mean(0)[None])
        avgs = torch.stack(avgs)

        key = self._avg_to_key(avgs)
        attention_output = self._attn(x, splits, key)

        # return as NxCxHxW
        return attention_output.reshape(attention_output.size(0),
                                        self.cfg.start_ch,
                                        self.cfg.start_width,
                                        self.cfg.start_width)


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
    def forward(self, x, splits, y):
        # (B x C x H x W) -pool-> (B x C)
        pooled = self._pool(y).view(y.size(0), -1)

        # (B x C) -proj-> (B x D)
        k = self._proj_k(pooled)
        v = self._proj_v(pooled)

        # (BS x D) -proj-> (BS x D)
        q = self._proj_q(x)

        # (BS x D) -split-> (BS x H x W)
        x, q, k, v = (split_last(x, (self.cfg.n_heads, -1))
                      for x in [x, q, k, v])

        out = []

        for i, (prev, after) in enumerate(splits):
            b_k = k[i].unsqueeze(0)
            b_v = v[i].unsqueeze(0)
            b_q = q[prev:after]
            b_x = x[prev:after]

            scores = (b_q[:, :, None] @ b_k[:, :, :, None]).squeeze(-1)
            scores = torch.sigmoid(scores)

            out.append(merge_last(b_x * (1 - scores) + b_v * scores, 2))

        # would probably also be reasonable to do just x + v * scores
        out = torch.cat(out, dim=0)

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

    # x is seq (BS x D), y is image type data (B x C x H x W)
    def forward(self, x, splits, y):
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
        added_values = self._attn(x, splits, k)
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
