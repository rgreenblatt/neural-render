import collections
import re
from distutils.util import strtobool
import math

# Parameters for the entire model
GlobalParams = collections.namedtuple('GlobalParams', [
    'start_width', 'end_width', 'input_size', 'input_expand_size',
    'nonlocal_index', 'norm_style', 'base_feature_extractor_args'
])

# Parameters for each model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'upsample', 'expand_ratio', 'input_ch',
    'output_ch', 'se_ratio', 'show_position', 'res', 'feature_extractor_args'
])

# Parameters for the input feature extraction
FeatureExtractorArgs = collections.namedtuple('FeatureExtractorArgs', [
    'has_channel_input', 'input_ch', 'mix_ch', 'embedding_size',
    'reduced_stats_size', 'expand_stat_embedding_size', 'output_size'
])


def net_params(width_coefficient=1.0,
               depth_coefficient=1.0,
               base_min_ch=16,
               start_width=4,
               input_size=3,
               input_expand_size=12,
               output_width=512,
               non_local_width=64,
               chan_multiplier=2,
               norm_style='bn'):
    """Create BlockArgs and GlobalParams

    Args:
        TODO
    Returns:
        blocks_args, global_params.
    """

    get_num_upsamples = lambda x: math.ceil(math.log2(x / start_width))

    num_upsamples = get_num_upsamples(output_width)
    nonlocal_index = get_num_upsamples(non_local_width)

    min_ch = base_min_ch * width_coefficient
    initial_ch = chan_multiplier**num_upsamples * min_ch
    num_repeat = 2  # TODO: better approach

    blocks_args = []

    ch_before = initial_ch

    res = start_width

    # TODO: tuning
    for i in range(num_upsamples):
        ch_after = ch_before / chan_multiplier
        input_ch = round(ch_before)
        output_ch = round(ch_after)
        blocks_args.append(
            BlockArgs(
                num_repeat=round(num_repeat * depth_coefficient),
                # TODO: does 5 x 5 improve things in some cases?
                kernel_size=3,
                # TODO: should some blocks just modify ch count?
                upsample=True,
                # TODO: tune
                expand_ratio=6,
                input_ch=input_ch,
                output_ch=output_ch,
                se_ratio=0.25,
                show_position=True,
                res=res,
                # all these numbers are super arbitrary
                feature_extractor_args=FeatureExtractorArgs(
                    has_channel_input=True,
                    input_ch=input_ch,
                    mix_ch=input_expand_size * 4,
                    embedding_size=8,
                    reduced_stats_size=32,
                    expand_stat_embedding_size=128,
                    output_size=input_expand_size * 2,
                )))
        ch_before = ch_after
        res *= 2

    base_extractor_out = round(initial_ch) * start_width**2

    global_params = GlobalParams(
        start_width=start_width,
        end_width=output_width,
        input_size=input_size,
        input_expand_size=input_expand_size,
        nonlocal_index=nonlocal_index,
        norm_style=norm_style,
        base_feature_extractor_args=FeatureExtractorArgs(
            has_channel_input=False,
            input_ch=None,
            mix_ch=base_extractor_out // 4,
            embedding_size=32,
            reduced_stats_size=128,
            expand_stat_embedding_size=265,
            output_size=base_extractor_out,
        ))

    return blocks_args, global_params
