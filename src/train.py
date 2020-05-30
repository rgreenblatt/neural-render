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
from utils import mkdirs, PiecewiseLinear, linear_to_srgb
from criterion import PerceptualLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-multiplier', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--valid-split-seed', type=int, default=0)
    parser.add_argument('--train-images-to-save', type=int, default=16)
    parser.add_argument('--test-images-to-save', type=int, default=16)
    parser.add_argument('--save-model-every', type=int, default=5)
    parser.add_argument('--name', required=True)
    parser.add_argument('--norm-style', default='bn')
    parser.add_argument('--max-ch', type=int, default=256)
    # parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--no-cudnn-benchmark', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--opt-level', default='O1')
    parser.add_argument('--no-sync-bn',
                        action='store_true',
                        help='do not use sync bn when running in parallel')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-len', type=int, default=4)
    parser.add_argument('--hide-model-info', action='store_true')
    parser.add_argument('--fake-data',
                        action='store_true',
                        help='disable loading data for profiling')

    parser.add_argument('--disable-seq-to-image', action='store_true')
    parser.add_argument('--no-perceptual-loss', action='store_true')
    parser.add_argument('--improve-seq', action='store_true')

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
        seq_size=512,
        initial_attn_ch=128,
        output_width=img_width,
        max_ch=args.max_ch,
        norm_style=args.norm_style,
        show_info=not hide_model_info,
        use_seq_to_image=not args.disable_seq_to_image,
        use_image_to_seq=args.improve_seq,
        use_seq_block=args.improve_seq,
    )

    net = Net(blocks_args, global_args)

    # does the order of these matter???
    if args.distributed and not args.no_sync_bn:
        net = convert_syncbn_model(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    net = net.to(device)

    # TODO: keep bn f32 arg?
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level)

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

    epoches = 100
    # epoch_mark_0 = 20
    # epoch_mark_1 = 40
    # epoch_mark_2 = 60
    # TODO: make this more configurable
    lr_schedule = PiecewiseLinear([
        (0, 5e-5),
        (4, 1e-4),
        (70, 5e-6),
        (100, 1e-7),
    ])

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

    train_batches_to_save = math.ceil(args.train_images_to_save / batch_size)
    test_batches_to_save = math.ceil(args.test_images_to_save / batch_size)

    initial_range_start = 0
    initial_range_end = -1

    train, test, epoch_callback = get_dataset(initial_range_start,
                                              initial_range_end)

    for epoch in range(epoches):
        net.train()
        train_loss = 0.0

        lr = args.lr_multiplier * lr_schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # if epoch == epoch_mark_0:
        #     print("extending dataset at epoch_mark_0")
        #     train, test, epoch_callback = get_dataset(0, -1)

        # if epoch == epoch_mark_1:
        #     print("extending dataset at epoch_mark_1")
        #     train, test, epoch_callback = get_dataset(0, 12288)

        # if epoch == epoch_mark_2:
        #     print("extending dataset at epoch_mark_2")
        #     train, test, epoch_callback = get_dataset(0, -1)

        epoch_callback(epoch)

        actual_images_train = None
        output_images_train = None

        for i, data in enumerate(train):
            inp = data['inp'].to(device)
            image = data['image'].to(device)

            optimizer.zero_grad()

            outputs = net(inp)

            if i < train_batches_to_save:
                cpu_images = linear_to_srgb(image.detach().cpu())
                cpu_outputs = linear_to_srgb(outputs.detach().cpu())
                if actual_images_train is not None:
                    actual_images_train = torch.cat(
                        (actual_images_train, cpu_images), dim=0)
                    output_images_train = torch.cat(
                        (output_images_train, cpu_outputs), dim=0)
                else:
                    actual_images_train = cpu_images
                    output_images_train = cpu_outputs

            loss = criterion(outputs, image)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            train_loss += loss.item()
            if args.profile and i >= args.profile_len:
                return

        train_loss /= len(train)

        net.eval()
        test_loss = 0.0

        actual_images_test = None
        output_images_test = None

        with torch.no_grad():
            for i, data in enumerate(test):
                inp = data['inp'].to(device)
                image = data['image'].to(device)

                outputs = net(inp)

                if i < test_batches_to_save:
                    cpu_images = linear_to_srgb(image.detach().cpu())
                    cpu_outputs = linear_to_srgb(outputs.detach().cpu())
                    if actual_images_test is not None:
                        actual_images_test = torch.cat(
                            (actual_images_test, cpu_images), dim=0)
                        output_images_test = torch.cat(
                            (output_images_test, cpu_outputs), dim=0)
                    else:
                        actual_images_test = cpu_images
                        output_images_test = cpu_outputs

                loss = criterion(outputs, image)

                test_loss += loss.item()

        test_loss /= len(test)

        if not disable_all_output:
            print(
                "{}, epoch: {}, train loss: {}, test loss: {}, lr: {}".format(
                    datetime.datetime.now(), epoch, train_loss, test_loss, lr))

            if actual_images_train is not None:
                writer.add_image("images/train/actual",
                                 make_grid(actual_images_train), epoch)
                writer.add_image("images/train/output",
                                 make_grid(output_images_train), epoch)

            if actual_images_test is not None:
                writer.add_image("images/test/actual",
                                 make_grid(actual_images_test), epoch)
                writer.add_image("images/test/output",
                                 make_grid(output_images_test), epoch)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("lr", lr, epoch)

        # if not disable_all_output and (epoch + 1) % args.save_model_every == 0:
        #     torch.save(
        #         net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    # if not disable_all_output:
    #     torch.save(net, os.path.join(model_save_output, "net_final.p"))


if __name__ == "__main__":
    main()
