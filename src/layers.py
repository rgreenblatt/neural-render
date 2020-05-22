import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter as P

from utils import (Swish, MemoryEfficientSwish)


# From big gan...
class ccbn(nn.Module):
    def __init__(
        self,
        output_size,
        input_size,
        which_linear,
        eps=1e-5,
        momentum=0.1,
        norm_style='bn',
    ):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Norm style?
        self.norm_style = norm_style

        if self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        if self.norm_style == 'bn':
            out = F.batch_norm(x, self.stored_mean, self.stored_var, None,
                               None, self.training, 0.1, self.eps)
        elif self.norm_style == 'in':
            out = F.instance_norm(x, self.stored_mean, self.stored_var, None,
                                  None, self.training, 0.1, self.eps)
        elif self.norm_style == 'gn':
            out = groupnorm(x, self.normstyle)
        elif self.norm_style == 'nonorm':
            out = x
        return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' norm_style={norm_style}'
        return s.format(**self.__dict__)


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
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args

        # pytorch's difference from tensorflow
        self.has_se = (self._block_args.se_ratio is
                       not None) and (0 < self._block_args.se_ratio <= 1)

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_ch  # number of input channels
        # number of output channels
        oup = self._block_args.input_ch * self._block_args.expand_ratio

        bn_linear = functools.partial(nn.Linear, bias=True)
        self.which_bn = functools.partial(
            ccbn,
            which_linear=bn_linear,
            input_size=block_args.feature_extractor_args.output_size,
            norm_style=global_params.norm_style)

        self._bn0 = self.which_bn(inp)

        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp,
                                          out_channels=oup,
                                          kernel_size=1,
                                          bias=False)
            self._bn1 = self.which_bn(oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            bias=False,
            padding=1)
        self._bn2 = self.which_bn(oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_ch * self._block_args.se_ratio))
            # TODO: this may not be efficient - use dense...
            # (probably doesn't matter)
            self._se_reduce = nn.Conv2d(in_channels=oup,
                                        out_channels=num_squeezed_channels,
                                        kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels,
                                        out_channels=oup,
                                        kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_ch
        self._project_conv = nn.Conv2d(in_channels=oup,
                                       out_channels=final_oup,
                                       kernel_size=1,
                                       bias=False)
        self._swish = MemoryEfficientSwish()

        self._upsample = functools.partial(F.interpolate, scale_factor=2)

    def forward(self, inputs, original_input_expanded, position_ch):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        x = self._bn0(x, original_input_expanded)
        x = self._swish(x)

        if self._block_args.show_position:
            expanded_pos = position_ch[self._block_args.res].expand(
                x.size(0), -1, -1, -1)

            x = torch.cat((x, expanded_pos), dim=1)

        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn1(x, original_input_expanded)
            x = self._swish(x)

        if self._block_args.upsample:
            x = self._upsample(x)
            inputs = self._upsample(inputs)[:, :self._block_args.output_ch]

        x = self._depthwise_conv(x)
        x = self._bn2(x, original_input_expanded)
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

        return x + inputs

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


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


# This probably needs a bunch of tunning
class FeatureExtractor(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args

        self._swish = MemoryEfficientSwish()

        if self._block_args.has_channel_input:
            self._avg_pool = nn.AdaptiveMaxPool2d(1)
            self._ch_to_mix = nn.Linear(self._block_args.input_ch,
                                        self._block_args.mix_ch,
                                        bias=False)
        self._input_to_mix = nn.Linear(global_params.input_expand_size,
                                       self._block_args.mix_ch,
                                       bias=True)
        self._embedding = nn.Linear(self._block_args.mix_ch,
                                    self._block_args.embedding_size,
                                    bias=True)
        self._reduce_stats_feature = nn.Linear(
            1 + 2 * (self._block_args.embedding_size +
                     self._block_args.embedding_size**2),
            self._block_args.reduced_stats_size,
            bias=True)
        self._expand_stat_embeddings = nn.Linear(
            self._block_args.reduced_stats_size +
            self._block_args.embedding_size,
            self._block_args.expand_stat_embedding_size,
            bias=True)
        self._to_multipliers = nn.Linear(
            self._block_args.expand_stat_embedding_size,
            self._block_args.output_size,
            bias=True)

        self._hidden = nn.Linear(self._block_args.mix_ch,
                                 self._block_args.output_size,
                                 bias=True)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

    def forward(self, inputs, splits, channels=None):

        if inputs.nelement() == 0:
            return torch.zeros((len(splits), self._block_args.output_size),
                               dtype=inputs.dtype,
                               device=inputs.device)

        x = self._input_to_mix(inputs)

        if self._block_args.has_channel_input:
            avg_ch = self._avg_pool(channels).view(channels.size(0), -1)
            mix_ch = self._ch_to_mix(avg_ch)
            mix_ch_resized = torch.cat([
                mix_ch[i].view(-1).expand(after - prev, -1)
                for i, (prev, after) in enumerate(splits)
            ])

            x = mix_ch_resized + x

        # TODO: consider adding bn (somewhat strange/not sure it would do what
        # I want)
        x = self._swish(x)

        embedding = self._embedding(x)

        stats = []

        all_products = embedding[:, None, :] * embedding[:, :, None]

        # may not be peak efficiency...
        for i, (prev, after) in enumerate(splits):
            count = after - prev
            if count == 0:
                continue

            count_t = torch.tensor([count],
                                   dtype=inputs.dtype,
                                   device=inputs.device).view(1, 1)
            b_embed = embedding[prev:after]

            # roughly expectation
            b_avgs = torch.mean(b_embed, dim=0, keepdim=True)

            # all combinations of products (for covariance matrix)
            b_all_products = all_products[prev:after].view(count, -1)

            # roughly covariance (as vector)
            b_all_products_avgs = torch.mean(b_all_products,
                                             dim=0,
                                             keepdim=True)

            stats_feature = torch.cat(
                (count_t, b_avgs, b_all_products_avgs, b_avgs * count_t,
                 b_all_products_avgs * count_t),
                dim=1)

            stats_feature_reduced = self._swish(
                self._reduce_stats_feature(stats_feature))

            stats.append(stats_feature_reduced.expand(count, -1))

        stats = torch.cat(stats, dim=0)
        expanded = self._expand_stat_embeddings(
            torch.cat((stats, embedding), dim=1))
        expanded = self._swish(expanded)
        multipliers = torch.sigmoid(self._to_multipliers(expanded))

        hidden_values = self._swish(self._hidden(x))

        pre_sum = multipliers * hidden_values

        out = []

        for i, (prev, after) in enumerate(splits):
            count = after - prev

            if count == 0:
                out.append(
                    torch.zeros((1, self._block_args.output_size),
                                dtype=inputs.dtype,
                                device=inputs.device))
            else:
                out.append(torch.mean(pre_sum[prev:after], dim=0, keepdim=True))

        out = torch.cat(out, dim=0)

        return out
