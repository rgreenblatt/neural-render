import torch
from torch import nn

from layers import (MBConvGBlock, Attention, Transformer, SequenceToImageStart,
                    SequenceToImage, ImageToSequence)
from utils import Swish, MemoryEfficientSwish, get_position_ch


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

        out_channels = 3  # rgb

        self._input_expand = nn.Linear(self._global_params.input_size,
                                       self._global_params.input_expand_size)

        # linear block
        self._base_transformer = Transformer(
            global_params.base_transformer_args())
        self._sequence_to_image_start = SequenceToImageStart(
            global_params.sequence_to_image_start_args())

        self._image_blocks = nn.ModuleList([])
        self._seq_blocks = nn.ModuleList([])
        self._seq_to_image_blocks = nn.ModuleList([])
        self._image_to_seq_blocks = nn.ModuleList([])

        last_ch = blocks_args[-1].output_ch

        for i, block_args in enumerate(blocks_args):
            input_ch = block_args.input_ch
            output_ch = block_args.output_ch

            for repeat_num in range(block_args.num_repeat):
                if repeat_num == block_args.num_repeat - 1:
                    block_args.last_block()
                self._image_blocks.append(
                    MBConvGBlock(block_args.mbconv_args(),
                                 self._global_params))
                self._image_to_seq_blocks.append(
                    ImageToSequence(block_args.image_to_seq_args()))
                self._seq_blocks.append(
                    Transformer(block_args.transformer_args()))
                self._seq_to_image_blocks.append(
                    SequenceToImage(block_args.seq_to_image_args()))

            if i == self._global_params.nonlocal_index - 1:
                self._attention_index = len(self._blocks) - 1
                self._attention = Attention(block_args.output_ch)

        self._swish = MemoryEfficientSwish()

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

        self._base_transformer.set_swish(memory_efficient)

        for block in self._image_blocks:
            block.set_swish(memory_efficient)

        for block in self._seq_blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs, splits):
        """
        Args:
            inputs (tensor): Input tensor.

        Returns:
            image output from values 0 to \infty
        """
        # NOTE!!! We use batch dimension in inputs to store each item,
        # so the number of items can vary.

        inputs_expanded = self._swish(self._input_expand(inputs))
        seq = self._base_transformer(inputs_expanded, splits)
        image = self._sequence_to_image_start(seq, splits)

        position_ch = get_position_ch(self._global_params.start_width,
                                      self._global_params.end_width,
                                      inputs.dtype, inputs.device)

        all_blocks = (self._image_blocks, self._image_to_seq_blocks,
                      self._seq_blocks, self._seq_to_image_blocks)

        for i, blocks in enumerate(zip(all_blocks)):
            image_b, image_to_seq_b, seq_b, seq_to_image_b = blocks
            image = image_b(image)
            seq = image_to_seq_b(seq, image)
            seq = seq_b(seq, splits, image)
            image = seq_to_image_b(seq, splits, image)

        image = self.output_bn(image)
        image = self._swish(image)
        image = self.output_conv(image)
        image = torch.exp(image)  # seems like a reasonable choice, but...

        return image
