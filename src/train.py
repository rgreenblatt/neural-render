import os
import argparse
import sys
import datetime
import math
import itertools

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid
from scipy.spatial.transform import Rotation as R
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from apex import amp

from model import Net
from load_data import load_dataset
from arch import net_params
from utils import (mkdirs, LRSched, linear_to_srgb, LossTracker, ImageTracker)
from criterion import PerceptualLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-multiplier', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--valid-split-seed', type=int, default=0)
    parser.add_argument('--train-images-to-save', type=int, default=64)
    parser.add_argument('--test-images-to-save', type=int, default=256)
    parser.add_argument('--save-model-every', type=int, default=5)
    parser.add_argument(
        '--display-freq',
        type=int,
        default=5000,
        help='number of samples per display print out and tensorboard save')
    parser.add_argument('--name', required=True)
    parser.add_argument('--norm-style', default='bn')
    parser.add_argument('--max-ch', type=int, default=256)
    parser.add_argument('--initial-attn-ch', type=int, default=128)
    parser.add_argument('--seq-size', type=int, default=512)
    parser.add_argument('--no-cudnn-benchmark', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--opt-level', default='O1')
    parser.add_argument('--no-sync-bn',
                        action='store_true',
                        help='do not use sync bn when running in parallel')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-len',
                        type=int,
                        default=5000,
                        help='number of samples to run for profiling')
    parser.add_argument('--hide-model-info', action='store_true')
    parser.add_argument('--fake-data',
                        action='store_true',
                        help='disable loading data for profiling')

    parser.add_argument('--no-seq-to-image', action='store_true')
    parser.add_argument('--no-perceptual-loss', action='store_true')
    parser.add_argument('--use-seq-blocks', action='store_true')
    parser.add_argument('--no-image-to-seq', action='store_true')
    parser.add_argument('--checkpoint-conv', action='store_true')
    parser.add_argument('--no-base-transformer', action='store_true')
    parser.add_argument('--only-descending-ch', action='store_true')
    parser.add_argument('--no-add-seq-to-image', action='store_true')
    parser.add_argument('--add-seq-to-image-mix-bias', type=float, default=0.0)
    parser.add_argument('--add-image-to-seq-mix-bias', type=float, default=-2.0)
    # ALSO TODO: no parameter sharing
    parser.add_argument('--base-transformer-n-layers', type=int, default=1)
    parser.add_argument('--seq-transformer-n-layers', type=int, default=1)
    parser.add_argument('--full-attn-ch', action='store_true')



    args = parser.parse_args()

    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    def reduce_tensor(tensor):
        if not args.distributed:
            return tensor

        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= args.world_size

        return rt

    device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.
                          is_available() else "cpu")

    data_path = 'data/'
    p_path = os.path.join(data_path, 'scenes.p')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = args.batch_size
    valid_prop = 0.2

    disable_all_output = args.local_rank != 0 or args.profile
    hide_model_info = args.hide_model_info or disable_all_output

    img_width = args.resolution

    # function for processing the input (shouldn't have trainable params)
    # takes np array
    def process_input(inp):
        # add many angle representations (not sure which is best yet)

        # switch from (W, X, Y, Z) to (X, Y, Z, W)
        quat_arr = np.concatenate((inp[:, 4:7], inp[:, 3:4]), axis=1)
        rot = R.from_quat(quat_arr)
        mat_arr = rot.as_matrix()
        mat_arr.resize((inp.shape[0], 9))
        rot_vec_arr = rot.as_rotvec()

        location = inp[:, :3]
        scale = inp[:, 7:10]

        cube_points = np.empty((inp.shape[0], 24))
        for i, point in enumerate(
                itertools.product([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])):
            point = np.tile(np.array(point), (inp.shape[0], 1))

            point *= scale
            point = rot.apply(point)
            point += location

            cube_points[:, i * 3:(i + 1) * 3] = point

        return np.concatenate((inp, mat_arr, rot_vec_arr, cube_points), axis=1)

    def get_dataset(start_range, end_range):
        return load_dataset(p_path,
                            img_path,
                            img_width,
                            batch_size,
                            valid_prop,
                            args.valid_split_seed,
                            True,
                            num_workers=8,
                            fake_data=args.fake_data,
                            process_input=process_input,
                            start_range=start_range,
                            end_range=end_range,
                            num_replicas=args.world_size,
                            rank=args.local_rank)

    input_size = 56  # 20, then 56 after process_input

    blocks_args, global_args = net_params(
        input_size=input_size,
        seq_size=args.seq_size,
        initial_attn_ch=args.initial_attn_ch,
        output_width=img_width,
        max_ch=args.max_ch,
        norm_style=args.norm_style,
        show_info=not hide_model_info,
        use_seq_to_image=not args.no_seq_to_image,
        use_image_to_seq=not args.no_image_to_seq,
        use_seq_block=args.use_seq_blocks,
        checkpoint_conv=args.checkpoint_conv,
        use_base_transformer=not args.no_base_transformer,
        only_descending_ch=args.only_descending_ch,
        add_seq_to_image=not args.no_add_seq_to_image,
        add_seq_to_image_mix_bias=args.add_seq_to_image_mix_bias,
        add_image_to_seq_mix_bias=args.add_image_to_seq_mix_bias,
        base_transformer_n_layers=args.base_transformer_n_layers,
        seq_transformer_n_layers=args.seq_transformer_n_layers,
        full_attn_ch=args.full_attn_ch
    )

    net = Net(blocks_args, global_args)

    # does the order of these matter???
    if args.distributed and not args.no_sync_bn:
        net = convert_syncbn_model(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    net = net.to(device)

    # TODO: keep bn f32 arg?
    net, optimizer = amp.initialize(net,
                                    optimizer,
                                    opt_level=args.opt_level,
                                    min_loss_scale=65536.0)

    if args.distributed:
        net = DistributedDataParallel(net)

    if not hide_model_info:
        print(net)

    def recursive_param_print(module, memo=None, value='', name="net"):
        if memo is None:
            memo = set()

        total = 0
        to_print = ""

        if module not in memo:
            memo.add(module)

            for m_name, c_module in module.named_children():
                child_total, child_to_print = recursive_param_print(
                    c_module, memo, "  " + value, name=m_name)

                total += child_total
                to_print += child_to_print

            for p_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    this_param_total = param.numel()
                    total += this_param_total
                    to_print += "  " + value + p_name + " (param) : {}\n".format(
                        this_param_total)

            if total != 0:
                to_print = value + name + ": {}\n".format(total) + to_print

        return total, to_print

    if not hide_model_info:
        print()

        total, to_print = recursive_param_print(net)
        print(to_print)

    if args.no_perceptual_loss:
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = PerceptualLoss().to(device)

    factor = 2e-6
    scaled_lr = args.lr_multiplier * batch_size * args.world_size * factor
    lr_schedule = LRSched(scaled_lr, args.epochs)

    if not disable_all_output:
        output_dir = "outputs/{}/".format(args.name)
        tensorboard_output = os.path.join(output_dir, "tensorboard")
        model_save_output = os.path.join(output_dir, "models")

        if os.path.exists(output_dir):
            print("output directory exists, returning")
            sys.exit(1)

        mkdirs(tensorboard_output)
        mkdirs(model_save_output)

        writer = SummaryWriter(log_dir=tensorboard_output)

    train_batches_save = math.ceil(args.train_images_to_save / batch_size)
    test_batches_save = math.ceil(args.test_images_to_save / batch_size)

    initial_range_start = 0
    initial_range_end = -1

    train, test, epoch_callback = get_dataset(initial_range_start,
                                              initial_range_end)

    step = 0

    world_batch_size = batch_size * args.world_size

    for epoch in range(args.epochs):
        epoch_callback(epoch)

        net.train()

        i = 0

        train_loss_tracker = LossTracker(reduce_tensor)

        lr = lr_schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        steps_since_display = 0
        max_train_step = (len(train) - 1) * world_batch_size
        format_len = math.floor(math.log10(max_train_step)) + 1

        def train_display():
            train_loss = train_loss_tracker.query_reset()
            if not disable_all_output:
                print("{}, epoch {}/{}, step {}/{}, train loss {:.4e}".format(
                    datetime.datetime.now(), epoch, args.epochs - 1,
                    str(i * world_batch_size).zfill(format_len),
                    max_train_step, train_loss))
                writer.add_scalar("loss/train", train_loss, step)

        if not disable_all_output:
            actual_images_train = ImageTracker()
            output_images_train = ImageTracker()

        for i, data in enumerate(train):
            inp = data['inp'].to(device)
            image = data['image'].to(device)

            step += world_batch_size
            steps_since_display += world_batch_size

            optimizer.zero_grad()

            outputs = net(inp)

            # save at end of training
            if not disable_all_output and len(train) - i <= train_batches_save:
                actual_images_train.update(image)
                output_images_train.update(outputs)

            loss = criterion(outputs, image)

            train_loss_tracker.update(loss)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if args.profile and step >= args.profile_len:
                return

            if steps_since_display >= args.display_freq:
                train_display()
                steps_since_display = 0

        if steps_since_display > 0:
            train_display()

        net.eval()

        test_loss_tracker = LossTracker(reduce_tensor)

        actual_images_test = ImageTracker()
        output_images_test = ImageTracker()

        with torch.no_grad():
            for i, data in enumerate(test):
                inp = data['inp'].to(device)
                image = data['image'].to(device)

                outputs = net(inp)

                if not disable_all_output and i < test_batches_save:
                    actual_images_test.update(image)
                    output_images_test.update(outputs)

                loss = criterion(outputs, image)

                test_loss_tracker.update(loss)

        test_loss = test_loss_tracker.query_reset()
        if not disable_all_output:
            print("{}, epoch {}/{}, lr {:.4e}, test loss {:.4e}".format(
                datetime.datetime.now(), epoch, args.epochs - 1, lr,
                test_loss))

            actual_images_train = actual_images_train.query_reset()
            output_images_train = output_images_train.query_reset()
            actual_images_test = actual_images_test.query_reset()
            output_images_test = output_images_test.query_reset()

            if actual_images_train is not None:
                writer.add_image("images/train/actual",
                                 make_grid(actual_images_train), step)
                writer.add_image("images/train/output",
                                 make_grid(output_images_train), step)

            if actual_images_test is not None:
                writer.add_image("images/test/actual",
                                 make_grid(actual_images_test), step)
                writer.add_image("images/test/output",
                                 make_grid(output_images_test), step)

            writer.add_scalar("loss/test", test_loss, step)
            writer.add_scalar("lr", lr, step)

        # if not disable_all_output and (epoch + 1) % args.save_model_every == 0:
        #     torch.save(
        #         net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    # if not disable_all_output:
    #     torch.save(net, os.path.join(model_save_output, "net_final.p"))


if __name__ == "__main__":
    main()
