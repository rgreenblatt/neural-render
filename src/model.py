import torch
from torch import nn

from layers import MBConvBlock, Attention
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

        out_channels = 3 # rgb

        # linear block
        self.linear = nn.Linear(
            global_params.input_size,
            self._blocks_args[0].input_ch * (global_params.start_width**2))

        self._blocks = nn.ModuleList([])

        for i, block_args in enumerate(self._blocks_args):
            for _ in range(block_args.num_repeat):
                self._blocks.append(MBConvBlock(blocks_args, self._global_params))

                blocks_args.upsample = False
                blocks_args.input_ch = blocks_args.output_ch

            if i == self._global_params.nonlocal_index-1:
                print("INDEX:", i)

                self._blocks.append(Attention(blocks_args.output_ch))

        last_ch = self._blocks_args[-1].output_ch

        self.output_bn = nn.BatchNorm2d(last_ch)
        self.output_conv = nn.Conv2d(last_ch, out_channels)

    def forward(self, inputs):
        """
        Args:
            inputs (tensor): Input tensor.

        Returns:
            image output from values 0 to 1
        """

        x = self.linear(inputs)
        for b in self._blocks:
            x = b(x, inputs) # later inputs here will be more complex?

        x = self.output_bn(x)
        x = self._swish(x)
        x = self.output_conv(x)
        x = torch.sigmoid(x)

        return x
