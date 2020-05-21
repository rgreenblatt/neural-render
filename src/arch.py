import collections
import re
from distutils.util import strtobool
import math

# Parameters for the entire model (TODO: non-local etc)
GlobalParams = collections.namedtuple('GlobalParams', [
    'start_width', 'end_width', 'input_size', 'input_expand_size',
    'nonlocal_index', 'norm_style'
])

# Parameters for each model block (TODO)
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'upsample', 'expand_ratio', 'input_ch',
    'output_ch', 'se_ratio', 'show_position', 'res'
])


def net_params(width_coefficient=1.0,
               depth_coefficient=1.0,
               base_min_ch=16,
               start_width=4,
               input_size=3,
               input_expand_size=15,
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
        blocks_args.append(
            BlockArgs(
                num_repeat=round(num_repeat * depth_coefficient),
                # TODO: does 5 x 5 improve things in some cases?
                kernel_size=3,
                # TODO: should some blocks just modify ch count?
                upsample=True,
                # TODO: tune
                expand_ratio=6,
                input_ch=round(ch_before),
                output_ch=round(ch_after),
                se_ratio=0.25,
                show_position=True,
                res=res,
            ))
        ch_before = ch_after
        res *= 2

    global_params = GlobalParams(start_width=start_width,
                                 end_width=output_width,
                                 input_size=input_size,
                                 input_expand_size=input_expand_size,
                                 nonlocal_index=nonlocal_index,
                                 norm_style=norm_style)

    return blocks_args, global_params
