import os
import argparse
import sys
import datetime
import math

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid

from model import Net
from load_data import load_dataset
from arch import net_params
from utils import mkdirs, PiecewiseLinear, linear_to_srgb


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
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--hide-model-info', action='store_true')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = 'data/'
    p_path = os.path.join(data_path, 'scenes.p')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = args.batch_size
    valid_prop = 0.2

    hide_model_info = args.hide_model_info or args.profile

    img_width = args.resolution

    train, test = load_dataset(p_path,
                               img_path,
                               img_width,
                               batch_size,
                               valid_prop,
                               args.valid_split_seed,
                               True,
                               num_workers=8)

    input_size = 20

    blocks_args, global_args = net_params(input_size=input_size,
                                          seq_size=128,
                                          output_width=img_width,
                                          max_ch=args.max_ch,
                                          norm_style=args.norm_style)

    net = Net(blocks_args, global_args).to(device)

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

    # TODO: consider switching to l1 (as opposed to l2)
    criterion = torch.nn.MSELoss()
    epoches = args.epoches
    lr_schedule = PiecewiseLinear([0, 10, 70, 100],
                                  [0.0005, 0.002, 0.0002, 0.00002])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    if not args.profile:
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

    for epoch in range(epoches):
        net.train()
        train_loss = 0.0

        lr = args.lr_multiplier * lr_schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        actual_images_train = None
        output_images_train = None

        for i, (splits, data) in enumerate(train):
            inp = data['inp'].to(device)
            image = data['image'].to(device)

            optimizer.zero_grad()

            outputs = net(inp, splits)

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
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if args.profile and i >= 32:
                return

        train_loss /= len(train)

        net.eval()
        test_loss = 0.0

        actual_images_test = None
        output_images_test = None

        with torch.no_grad():
            for i, (splits, data) in enumerate(test):
                inp = data['inp'].to(device)
                image = data['image'].to(device)

                outputs = net(inp, splits)

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

        print("{}, epoch: {}, train loss: {}, test loss: {}, lr: {}".format(
            datetime.datetime.now(), epoch, train_loss, test_loss, lr))

        if not args.profile:
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

        if not args.profile and (epoch + 1) % args.save_model_every == 0:
            torch.save(
                net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    if not args.profile:
        torch.save(net, os.path.join(model_save_output, "net_final.p"))


if __name__ == "__main__":
    main()
