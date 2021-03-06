import argparse


class Config(argparse.Namespace):
    def ordered_params(self):
        return sorted(vars(self).items())

    def ordered_non_default(self):
        all_defaults = {
            # rebuild parser to avoiding having parser attribute
            key: self.build_parser().get_default(key)
            for key in vars(self)
        }

        def arg_is_default(attr, value):
            return all_defaults[attr] == value

        return map(
            lambda x: x + (all_defaults[x[0]], ),
            filter(lambda x: not arg_is_default(*x), self.ordered_params()))

    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in self.ordered_params():
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def print_non_default(self, prtf=print):
        prtf("")
        prtf("Non default parameters:")
        for attr, value, default in self.ordered_non_default():
            prtf("{}={} (default={})".format(attr.upper(), value, default))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in self.ordered_params():
            text += "|{}|{}|  \n".format(attr, value)

        return text

    def non_default_as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|default|  \n|-|-|-|  \n"
        for attr, value, default in self.ordered_non_default():
            text += "|{}|{}|{}|  \n".format(attr, value, default)

        return text

    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0)

        parser.add_argument('--lr-multiplier', type=float, default=1.0)
        parser.add_argument('--no-perceptual-loss', action='store_true')
        parser.add_argument('--batch-size', type=int, default=16)
        parser.add_argument('--no-cudnn-benchmark', action='store_true')
        parser.add_argument('--no-fused-adam', action='store_true')
        parser.add_argument('--opt-level', default='O0')
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--resolution', type=int, default=128)
        parser.add_argument('--valid-split-seed', type=int, default=0)
        parser.add_argument('--start-max-seq-len', type=int, default=None)
        parser.add_argument('--start-min-prop-emissive',
                            type=float,
                            default=None)
        parser.add_argument('--min-prop-emissive-change-rate',
                            type=float,
                            default=0.2)
        parser.add_argument('--change-factors-freq', type=int, default=2)
        parser.add_argument('--data-count-limit', type=int, default=None)
        parser.add_argument('--amp-verbosity', type=int, default=0)
        parser.add_argument('--train-images-to-save', type=int, default=64)
        parser.add_argument('--test-images-to-save', type=int, default=256)
        parser.add_argument('--save-model-every', type=int, default=5)
        parser.add_argument(
            '--display-freq',
            type=int,
            default=5000,
            help='number of samples per display print out and tensorboard save'
        )
        parser.add_argument('--set-lr-freq',
                            type=int,
                            default=5000,
                            help='number of samples per setting optimizer lr')
        parser.add_argument('--show-model-info', action='store_true')
        parser.add_argument('--name', required=True)

        parser.add_argument('--profile', action='store_true')
        parser.add_argument('--profile-len',
                            type=int,
                            default=5000,
                            help='number of samples to run for profiling')
        parser.add_argument('--fake-data',
                            action='store_true',
                            help='disable loading data for profiling')

        parser.add_argument('--norm-style', default='bn')
        parser.add_argument('--start-ch', type=int, default=256)
        parser.add_argument('--end-linear-ch', type=int, default=256)
        parser.add_argument('--start-width', type=int, default=4)
        parser.add_argument('--seq-size', type=int, default=512)
        parser.add_argument('--no-sync-bn',
                            action='store_true',
                            help='do not use sync bn when running in parallel')
        parser.add_argument('--use-nonlocal', action='store_true')
        parser.add_argument('--nonlocal-width', type=int, default=64)
        parser.add_argument('--no-seq-to-image', action='store_true')
        parser.add_argument('--no-image-to-seq', action='store_true')
        parser.add_argument('--no-seq-blocks', action='store_true')
        parser.add_argument('--checkpoint-conv', action='store_true')
        parser.add_argument('--no-base-transformer', action='store_true')
        parser.add_argument('--add-seq-to-image-mix-bias',
                            type=float,
                            default=0.0)
        parser.add_argument('--add-image-to-seq-mix-bias',
                            type=float,
                            default=-7.0)
        # ALSO TODO: no parameter sharing
        parser.add_argument('--base-transformer-n-layers', type=int, default=4)
        parser.add_argument('--seq-transformer-n-layers', type=int, default=2)
        parser.add_argument('--attn-ch-frac', type=float, default=0.5)
        parser.add_argument('--key-ch-multip', type=float, default=0.25)
        parser.add_argument('--image-ch-per-head', type=float, default=64)
        parser.add_argument('--no-seq-to-image-start-use-feat-to-output',
                            action='store_true')
        parser.add_argument('--full-seq-frequency', action='store_true')
        parser.add_argument('--no-position-ch', action='store_true')
        parser.add_argument('--use-se', action='store_true')
        parser.add_argument('--linear-ch-blocks', type=int, default=2)

        parser.add_argument('--ch-coefficient', type=float, default=1.0)
        parser.add_argument('--depth-coefficient', type=float, default=1.0)
        parser.add_argument('--attn-excitation', action='store_true')
        parser.add_argument('--transformer-param-sharing', action='store_true')
        parser.add_argument('--transformer-cross-attn', action='store_true')
        parser.add_argument('--apply-at-expand', action='store_true')
        parser.add_argument('--key-ratio', type=float, default=1.0)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
