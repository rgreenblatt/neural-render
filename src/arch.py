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
    'start_width',
    'end_width',
    'input_size',
    'seq_size',
    'base_transformer_n_heads',
    'base_transformer_n_layers',
    'nonlocal_index',
    'use_nonlocal',
    'start_ch',
    'ch_per_head',
    'norm_style',
    'checkpoint_conv',
    'use_base_transformer',
    'seq_to_image_start_use_feat_to_output',
])


class GlobalArgs(_GlobalArgParams):
    def base_transformer_args(self):
        return TransformerCfg(size=self.seq_size,
                              n_heads=self.base_transformer_n_heads,
                              n_layers=self.base_transformer_n_layers,
                              hidden_ff_size=self.seq_size * 4)

    def seq_to_image_start_args(self):
        return subset_named_tuple(
            SeqToImageStartCfg,
            self,
            use_feat_to_output=self.seq_to_image_start_use_feat_to_output)


# Parameters for each model block
_BlockArgsParams = collections.namedtuple('BlockArgsParams', [
    'num_repeat',
    'kernel_size',
    'upsample',
    'expand_ratio',
    'input_ch',
    'output_ch',
    'se_ratio',
    'use_position_ch',
    'seq_size',
    'attn_ch',
    'seq_n_heads',
    'transformer_n_layers',
    'image_n_heads',
    'norm_style',
    'round_by',
    'show_info',
    'use_seq_to_image',
    'use_image_to_seq',
    'use_seq_block',
    'add_seq_to_image',
    'add_seq_to_image_mix_bias',
    'add_image_to_seq_mix_bias',
    'full_seq_frequency',
    'alternate_seq_block',
    'attn_excitation',
    'continuously_vary_ch',
])


class BlockArgs(_BlockArgsParams):
    def next_block(self):
        # basically __init__ hack because of issues with named tuple
        if not hasattr(self, 'block_num'):
            self.block_num = 0
            self.output_ch_this_block = self.input_ch
            if self.continuously_vary_ch:
                self.ch_per_block = (self.output_ch -
                                     self.input_ch) / self.num_repeat
            else:
                self.ch_per_block = 0.0

        is_first_block = self.block_num == 0
        is_last_block = self.block_num == self.num_repeat - 1

        self.upsample_this_block = is_last_block
        self.input_ch_this_block = self.output_ch_this_block
        self.output_ch_this_block = (self.input_ch_this_block +
                                     self.ch_per_block)
        if is_last_block:
            self.output_ch_this_block = self.output_ch

        # this could change...
        # we use sequence blocks on the first block
        self.use_image_to_seq_this_block = (self.use_seq_to_image
                                            and (is_first_block
                                                 or self.full_seq_frequency))
        self.use_seq_this_block = (self.use_seq_block and
                                   (is_first_block or self.full_seq_frequency))

        # and reincorporate sequence on the last block
        self.use_seq_to_image_this_block = (self.use_image_to_seq
                                            and (is_last_block
                                                 or self.full_seq_frequency))

        def round_valid(value):
            return math.ceil(round(value) / self.round_by) * self.round_by

        self.input_ch_conv = round_valid(self.input_ch_this_block)
        self.output_ch_conv = round_valid(self.output_ch_this_block)

        if self.use_seq_to_image_this_block and not self.add_seq_to_image:
            self.output_ch_conv -= round(self.attn_ch)

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
                             n_heads=self.seq_n_heads,
                             mix_bias=self.add_image_to_seq_mix_bias)

    def transformer_args(self):
        # hidden_ff_size could be configured further
        return TransformerCfg(size=self.seq_size,
                              n_heads=self.seq_n_heads,
                              n_layers=self.transformer_n_layers,
                              hidden_ff_size=self.seq_size * 4)

    def seq_to_image_args(self):
        return SeqToImageCfg(image_ch=self.output_ch_conv,
                             seq_size=self.seq_size,
                             output_ch=self.attn_ch,
                             n_heads=self.image_n_heads,
                             add_all_ch=self.add_seq_to_image,
                             mix_bias=self.add_seq_to_image_mix_bias)


def net_params(input_size, output_width, cfg):
    """Create BlockArgs and GlobalParams

    Args:
        TODO
    Returns:
        blocks_args, global_params.
    """

    get_num_upsamples = lambda x: math.ceil(math.log2(x / cfg.start_width))

    num_upsamples = get_num_upsamples(output_width)
    nonlocal_index = get_num_upsamples(cfg.nonlocal_width)

    ch_per_linear = (cfg.end_linear_ch - cfg.start_ch) / cfg.linear_ch_blocks

    # right now, ch_coefficient only effects start_ch, so it is effectively
    # not needed.
    start_ch = cfg.start_ch * cfg.ch_coefficient
    num_repeat = round(2 * cfg.depth_coefficient)  # TODO: better approach

    blocks_args = []

    input_ch = start_ch
    output_ch = None

    assert not cfg.norm_style.startswith('gn'), "group norm not supported"

    width = cfg.start_width

    # TODO: tuning
    for i in range(num_upsamples):
        is_linear = i < cfg.linear_ch_blocks
        if is_linear:
            output_ch = input_ch
        else:
            output_ch = input_ch / 2

        width *= 2

        attn_ch = output_ch * cfg.attn_ch_frac

        if cfg.show_model_info:
            print("{}: input: {}, output: {}, attn: {}, width: {}, repeat: {}".
                  format(i, input_ch, output_ch, attn_ch, width, num_repeat))

        expand_ratio = 6

        blocks_args.append(
            BlockArgs(
                num_repeat=num_repeat,
                # TODO: is 5x5 worthwhile?
                kernel_size=3,
                upsample=True,
                # TODO: tune
                expand_ratio=expand_ratio,
                input_ch=round(input_ch),
                output_ch=round(output_ch),
                se_ratio=0.25 if cfg.use_se else None,
                use_position_ch=not cfg.no_position_ch,
                seq_size=cfg.seq_size,
                attn_ch=round(attn_ch),
                seq_n_heads=8,
                transformer_n_layers=cfg.seq_transformer_n_layers,
                image_n_heads=2,  # TODO: test making fixed per...
                norm_style=cfg.norm_style,
                round_by=8,
                show_info=cfg.show_model_info,
                use_seq_to_image=not cfg.no_seq_to_image,
                use_image_to_seq=not cfg.no_image_to_seq,
                use_seq_block=cfg.use_seq_blocks,
                add_seq_to_image=not cfg.no_add_seq_to_image,
                add_seq_to_image_mix_bias=cfg.add_seq_to_image_mix_bias,
                add_image_to_seq_mix_bias=cfg.add_image_to_seq_mix_bias,
                full_seq_frequency=cfg.full_seq_frequency,
                alternate_seq_block=cfg.alternate_seq_block,
                attn_excitation=cfg.attn_excitation,
                continuously_vary_ch=is_linear,
            ))
        input_ch = output_ch

    global_args = GlobalArgs(
        start_width=cfg.start_width,
        end_width=output_width,
        input_size=input_size,
        seq_size=cfg.seq_size,
        base_transformer_n_heads=8,
        base_transformer_n_layers=cfg.base_transformer_n_layers,
        nonlocal_index=nonlocal_index,
        use_nonlocal=cfg.use_nonlocal,
        start_ch=round(start_ch),
        norm_style=cfg.norm_style,
        checkpoint_conv=cfg.checkpoint_conv,
        use_base_transformer=not cfg.no_base_transformer,
        seq_to_image_start_use_feat_to_output=not cfg.
        no_seq_to_image_start_use_feat_to_output,
        ch_per_head=32,
    )

    return blocks_args, global_args
