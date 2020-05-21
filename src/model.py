import torch
from torch import nn

from layers import MBConvGBlock, Attention
from utils import Swish, MemoryEfficientSwish


class Net(nn.Module):
    """The model

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    """
    def __init__(self, blocks_args, global_params):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        out_channels = 3  # rgb

        # linear block
        self.linear = nn.Linear(
            self._global_params.input_size, self._blocks_args[0].input_ch *
            (self._global_params.start_width**2))

        self._blocks = nn.ModuleList([])

        for i, block_args in enumerate(self._blocks_args):
            input_ch = block_args.input_ch
            output_ch = block_args.output_ch

            block_args = block_args._replace(upsample=False,
                                             output_ch=input_ch)
            for _ in range(block_args.num_repeat - 1):

                self._blocks.append(
                    MBConvGBlock(block_args, self._global_params))

            block_args = block_args._replace(upsample=True,
                                             output_ch=output_ch)

            self._blocks.append(MBConvGBlock(block_args, self._global_params))

            if i == self._global_params.nonlocal_index - 1:
                self._blocks.append(Attention(block_args.output_ch))

        self._swish = MemoryEfficientSwish()

        last_ch = self._blocks_args[-1].output_ch

        self.output_bn = nn.BatchNorm2d(last_ch)
        self.output_conv = nn.Conv2d(last_ch,
                                     out_channels,
                                     kernel_size=3,
                                     padding=1)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs):
        """
        Args:
            inputs (tensor): Input tensor.

        Returns:
            image output from values 0 to 1
        """

        x = self.linear(inputs)
        x = x.view(x.size(0), -1, self._global_params.start_width,
                   self._global_params.start_width)
        for b in self._blocks:
            x = b(x, inputs)  # later inputs here will be more complex?

        x = self.output_bn(x)
        x = self._swish(x)
        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x
