import sys
import argparse

from gen_utils import random_seed


class BaseConfig(argparse.Namespace):
    def build_base_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--resolution', type=int, default=1024)
        parser.add_argument('--samples', type=int, default=128)
        parser.add_argument('--count', type=int, default=128)
        parser.add_argument('--no-gpu', action='store_true')

        return parser

    def base_as_arg_string(self):
        # Not a very clean approach...
        return "--resolution {} --samples {} --count {} {}".format(
            self.resolution,
            self.samples,
            self.count,
            "--no-gpu" if self.no_gpu else "",
        )

    def __init__(self, argv=None):
        parser = self.build_parser()
        if argv is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv)
        super().__init__(**vars(args))


class GenerateConfig(BaseConfig):
    def build_parser(self):
        parser = self.build_base_parser()

        parser.add_argument("--seed", type=int, default=0)

        return parser


class GenerateUploadConfig(BaseConfig):
    def build_parser(self):
        parser = self.build_base_parser()

        parser.add_argument("api_key")
        parser.add_argument("--base-seed", type=int, default=random_seed())
        parser.add_argument("--app-dir", default="sphere_renders")

        return parser