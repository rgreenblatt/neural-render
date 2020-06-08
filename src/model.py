import torch
from torch import nn

from layers import (MBConvGBlock, Attention, Transformer, SeqToImageStart,
                    SeqToImage, ImageToSeq, ConfigurableNorm)
from utils import Swish, MemoryEfficientSwish, get_position_ch


class Net(nn.Module):
    """The model

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_args (namedtuple): Args for the entire model
    """
    def __init__(self, blocks_args, global_args):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_args = global_args

        out_channels = 3  # rgb

        self._input_expand = nn.Linear(self._global_args.input_size,
                                       self._global_args.seq_size)


        def get_block(constructor, use, *args):
            if use:
                return constructor(*args)
            else:
                return None

        # linear block
        self._base_transformer = get_block(
            Transformer, self._global_args.use_base_transformer,
            self._global_args.base_transformer_args())
        self._seq_to_image_start = SeqToImageStart(
            self._global_args.seq_to_image_start_args())

        self._image_blocks = nn.ModuleList([])
        self._image_to_seq_blocks = nn.ModuleList([])
        self._seq_blocks = nn.ModuleList([])
        self._seq_to_image_blocks = nn.ModuleList([])

        last_ch = blocks_args[-1].output_ch

        for i, block_args in enumerate(blocks_args):
            input_ch = block_args.input_ch
            output_ch = block_args.output_ch

            for repeat_num in range(block_args.num_repeat):
                block_args.next_block()

                # TODO: consider not constructing blocks which aren't used...
                self._image_blocks.append(
                    MBConvGBlock(block_args.mbconv_args()))
                self._image_to_seq_blocks.append(
                    get_block(ImageToSeq, block_args.use_image_to_seq_this_block,
                              block_args.image_to_seq_args()))
                self._seq_blocks.append(
                    get_block(Transformer, block_args.use_seq_this_block,
                              block_args.transformer_args()))
                self._seq_to_image_blocks.append(
                    get_block(SeqToImage, block_args.use_seq_to_image_this_block,
                              block_args.seq_to_image_args()))

            if (self._global_args.use_nonlocal
                    and i == self._global_args.nonlocal_index - 1):
                self._attention_index = len(self._image_blocks) - 1
                self._attention = Attention(block_args.output_ch)

        self._swish = MemoryEfficientSwish()

        self.output_bn = ConfigurableNorm(
            last_ch,
            input_gain_bias=False,
            norm_style=self._global_args.norm_style)
        self.output_conv = nn.Conv2d(last_ch,
                                     out_channels,
                                     kernel_size=3,
                                     padding=1)

        negative_allowance = 0.05

        # CELU might be a good choice...
        self._output_activation = nn.CELU(alpha=negative_allowance)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

        def block_set_swish(block):
            if block is not None:
                block.set_swish(memory_efficient)

        block_set_swish(self._base_transformer)

        for block in self._image_blocks:
            block_set_swish(block)

        for block in self._seq_blocks:
            block_set_swish(block)

    def reset_running_stats(self):
        self.output_bn.reset_running_stats()

        for block in self._image_blocks:
            block.reset_running_stats()


    def forward(self, inputs, masks, counts):
        """
        Args:
            inputs (tensor): Input tensor.

        Returns:
            image output from values 0 to \infty
        """
        # NOTE!!! We use batch dimension in inputs to store each item,
        # so the number of items can vary.

        # consider layernorm here...
        seq = self._swish(self._input_expand(inputs))

        if self._base_transformer is not None:
            seq = self._base_transformer(seq, masks, counts)
        image = self._seq_to_image_start(seq, masks, counts)

        position_ch = get_position_ch(self._global_args.start_width,
                                      self._global_args.end_width,
                                      inputs.dtype, inputs.device)

        all_blocks = (self._image_blocks, self._image_to_seq_blocks,
                      self._seq_blocks, self._seq_to_image_blocks)

        for i, blocks in enumerate(zip(*all_blocks)):
            (image_b, image_to_seq_b, seq_b, seq_to_image_b) = blocks

            this_position_ch = position_ch[image.size(2)]

            if self._global_args.checkpoint_conv:
                image = torch.utils.checkpoint.checkpoint(
                    image_b, image, this_position_ch)
            else:
                image = image_b(image, this_position_ch)

            if image_to_seq_b is not None:
                seq = image_to_seq_b(seq, image)
            if seq_b is not None:
                seq = seq_b(seq, masks, counts)
            if seq_to_image_b is not None:
                image = seq_to_image_b(seq, masks, counts, image)

            if self._global_args.use_nonlocal and i == self._attention_index:
                image = self._attention(image)

        image = self.output_bn(image)
        image = self._swish(image)
        image = self.output_conv(image)
        image = self._output_activation(image)

        return image
