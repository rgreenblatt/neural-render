import collections
import inspect
import re
from distutils.util import strtobool
import math

from layers import (MBConvCfg, ImageToSeqCfg, TransformerCfg, SeqToImageCfg,
                    SeqToImageStartCfg)

# TODO:
#  - consider making seq vary over the entire model
#  - consider making key size something different


def subset_named_tuple(to_tup, from_tup, **kwargs):
    args = inspect.getargspec(to_tup).args[1:]

    argdict = dict(from_tup._asdict())

    args = set(args).intersection(set(argdict.keys()))

    return to_tup(**{**{k: argdict[k] for k in args}, **kwargs})


# Parameters for the entire model
_GlobalArgParams = collections.namedtuple('GlobalArgsParams', [
    'start_width', 'end_width', 'input_size', 'seq_size',
    'base_transformer_n_heads', 'base_transformer_n_layers', 'nonlocal_index',
    'start_ch', 'ch_per_head', 'norm_style'
])


class GlobalArgs(_GlobalArgParams):
    def base_transformer_args(self):
        return TransformerCfg(size=self.seq_size,
                              n_heads=self.base_transformer_n_heads,
                              n_layers=self.base_transformer_n_layers,
                              hidden_ff_size=self.seq_size * 4)

    def seq_to_image_start_args(self):
        return subset_named_tuple(SeqToImageStartCfg, self)


# Parameters for each model block
_BlockArgsParams = collections.namedtuple('BlockArgsParams', [
    'num_repeat',
    'kernel_size',
    'upsample',
    'expand_ratio',
    'input_ch',
    'output_ch',
    'se_ratio',
    'show_position',
    'seq_size',
    'attn_output_ch',
    'seq_n_heads',
    'transformer_n_layers',
    'image_n_heads',
    'norm_style',
    'round_by',
    'show_info'
])


class BlockArgs(_BlockArgsParams):
    def next_block(self):
        # basically __init__ hack because of issues with named tuple
        if not hasattr(self, 'block_num'):
            self.block_num = 0
            self.output_ch_this_block = self.input_ch
            self.ch_per_block = (self.output_ch -
                                 self.input_ch) / self.num_repeat

        self.upsample_this_block = self.block_num == self.num_repeat - 1
        self.input_ch_this_block = self.output_ch_this_block
        self.output_ch_this_block = (self.input_ch_this_block +
                                     self.ch_per_block)
        def round_valid(value):
            return math.ceil(round(value) / self.round_by) * self.round_by

        self.input_ch_conv = round_valid(self.input_ch_this_block)
        self.output_ch_conv = (round_valid(self.output_ch_this_block) -
                               round(self.attn_output_ch))

        if self.show_info:
            print("block num:", self.block_num)
            print("input conv at n:", self.input_ch_conv)
            print("output conv at n:", self.output_ch_conv)
            print("output overall at n:", self.output_ch_this_block)
            print("output at end of blocks:", self.output_ch)

        self.block_num += 1

    def mbconv_args(self):
        return subset_named_tuple(MBConvCfg,
                                  self,
                                  upsample=self.upsample_this_block,
                                  input_ch=self.input_ch_conv,
                                  output_ch=self.output_ch_conv)

    def image_to_seq_args(self):
        # n_heads could be configured further
        return ImageToSeqCfg(image_ch=self.output_ch_conv,
                             seq_size=self.seq_size,
                             n_heads=self.seq_n_heads)

    def transformer_args(self):
        # hidden_ff_size could be configured further
        return TransformerCfg(size=self.seq_size,
                              n_heads=self.seq_n_heads,
                              n_layers=self.transformer_n_layers,
                              hidden_ff_size=self.seq_size * 4)

    def seq_to_image_args(self):
        return SeqToImageCfg(image_ch=self.output_ch_conv,
                             seq_size=self.seq_size,
                             output_ch=self.attn_output_ch,
                             n_heads=self.image_n_heads)


def net_params(input_size,
               seq_size,
               output_width,
               ch_coefficient=1.0,
               depth_coefficient=1.0,
               start_width=4,
               non_local_width=64,
               start_ch=128,
               start_ch_per_head=32,
               max_ch=512,
               chan_reduce_multiplier=2,
               norm_style='bn',
               show_info=True):
    """Create BlockArgs and GlobalParams

    Args:
        TODO
    Returns:
        blocks_args, global_params.
    """

    get_num_upsamples = lambda x: math.ceil(math.log2(x / start_width))

    num_upsamples = get_num_upsamples(output_width)
    nonlocal_index = get_num_upsamples(non_local_width)

    start_ch *= ch_coefficient
    max_ch *= ch_coefficient

    # TODO: change this
    increase_upsamples = num_upsamples // 2
    ch_per_increase = (max_ch - start_ch) / increase_upsamples

    num_repeat = 2  # TODO: better approach

    blocks_args = []

    ch_before = start_ch

    initial_attn_ch = seq_size // 4
    attn_ch = initial_attn_ch

    # TODO: tuning
    for i in range(num_upsamples):
        if i < increase_upsamples:
            ch_after = ch_before + ch_per_increase
        else:
            ch_after = ch_before / chan_reduce_multiplier
            attn_ch /= chan_reduce_multiplier


        input_ch = round(ch_before)
        output_ch = round(ch_after)


        if show_info:
            print("Layer {} input ch: {}, output ch: {}, attn ch: {}".format(
                i, input_ch, output_ch, attn_ch))

        expand_ratio = 6

        blocks_args.append(
            BlockArgs(
                num_repeat=round(num_repeat * depth_coefficient),
                # TODO: does 5 x 5 improve things in some cases?
                kernel_size=3,
                # TODO: should some blocks just modify ch count (we interpolate
                # ch changes regardless)?
                upsample=True,
                # TODO: tune
                expand_ratio=expand_ratio,
                input_ch=input_ch,
                output_ch=output_ch,
                se_ratio=0.25,
                show_position=True,
                seq_size=seq_size,
                attn_output_ch=round(attn_ch),
                seq_n_heads=8,
                transformer_n_layers=2,
                image_n_heads=2,
                norm_style=norm_style,
                # TODO: fix hack
                round_by=16 if norm_style.startswith('gn') else 1,
                show_info=show_info,
            ))

        ch_before = ch_after

    global_args = GlobalArgs(
        start_width=start_width,
        end_width=output_width,
        input_size=input_size,
        seq_size=seq_size,
        base_transformer_n_heads=8,
        base_transformer_n_layers=4,
        nonlocal_index=nonlocal_index,
        start_ch=round(start_ch),
        ch_per_head=start_ch_per_head,
        norm_style=norm_style,
    )

    return blocks_args, global_args
