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
    'has_channel_input', 'input_ch', 'mix_ch', 'initial_embedding_size',
    'reduced_stats_size', 'embedding_info_size', 'linear_sum_size',
    'linear_output_size'
])


def net_params(input_size,
               input_expand_size,
               output_width,
               width_coefficient=1.0,
               depth_coefficient=1.0,
               start_width=4,
               non_local_width=64,
               initial_ch=32,
               max_ch=512,
               chan_reduce_multiplier=2,
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

    initial_ch = 32
    # TODO: change this
    increase_upsamples = num_upsamples // 2
    ch_per_increase = (max_ch - initial_ch) / increase_upsamples

    num_repeat = 2  # TODO: better approach

    blocks_args = []

    ch_before = initial_ch

    res = start_width

    # TODO: tuning
    for i in range(num_upsamples):
        if i < increase_upsamples:
            ch_after = ch_before + ch_per_increase
        else:
            ch_after = ch_before / chan_reduce_multiplier

        input_ch = round(ch_before)
        output_ch = round(ch_after)

        print("Layer {} input ch: {}, output ch".format(
            i, input_ch, output_ch))

        expand_ratio = 6

        blocks_args.append(
            BlockArgs(
                num_repeat=round(num_repeat * depth_coefficient),
                # TODO: does 5 x 5 improve things in some cases?
                kernel_size=3,
                # TODO: should some blocks just modify ch count?
                upsample=True,
                # TODO: tune
                expand_ratio=expand_ratio,
                input_ch=input_ch,
                output_ch=output_ch,
                se_ratio=0.25,
                show_position=True,
                res=res,
                # all these numbers are super arbitrary
                feature_extractor_args=FeatureExtractorArgs(
                    has_channel_input=True,
                    input_ch=input_ch,
                    mix_ch=input_expand_size,
                    initial_embedding_size=8,
                    reduced_stats_size=32,
                    embedding_info_size=16,
                    linear_sum_size=input_expand_size * 2,
                    # This is for all the gains and biases in the first layer
                    linear_output_size=2 * input_ch,
                    # TODO: args for conv part
                )))
        ch_before = ch_after
        res *= 2

    base_extractor_out = initial_ch * start_width**2

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
            mix_ch=input_size,
            initial_embedding_size=8,
            reduced_stats_size=32,
            embedding_info_size=16,
            linear_sum_size=2 * input_size,
            linear_output_size=base_extractor_out,
        ))

    return blocks_args, global_params
