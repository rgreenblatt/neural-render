import os
import argparse
import sys
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid

from model import Net
from load_data import load_dataset, linear_to_srgb
from arch import net_params
from utils import mkdirs, PiecewiseLinear


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
    parser.add_argument('--norm-type', default='bn')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = 'data/'
    p_path = os.path.join(data_path, 'scenes.p')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = args.batch_size
    valid_prop = 0.2

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

    blocks_args, global_params = net_params(base_min_ch=32,
                                            output_width=img_width,
                                            input_size=input_size,
                                            input_expand_size=4 * input_size,
                                            norm_type=args.norm_type)

    net = Net(blocks_args, global_params).to(device)

    print(net)

    # TODO: consider switching to l1 (as opposed to l2)
    criterion = torch.nn.MSELoss()
    epoches = 100
    lr_schedule = PiecewiseLinear([0, 10, 70, 100],
                                  [0.0005, 0.002, 0.0002, 0.00002])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    output_dir = "outputs/{}/".format(args.name)
    tensorboard_output = os.path.join(output_dir, "tensorboard")
    model_save_output = os.path.join(output_dir, "models")

    if os.path.exists(output_dir):
        print("output directory exists, returning")
        sys.exit(1)

    mkdirs(tensorboard_output)
    mkdirs(model_save_output)

    writer = SummaryWriter(log_dir=tensorboard_output)

    train_batches_to_save = args.train_images_to_save // batch_size
    test_batches_to_save = args.test_images_to_save // batch_size

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
        writer.add_image("images/train/actual", make_grid(actual_images_train),
                         epoch)
        writer.add_image("images/train/output", make_grid(output_images_train),
                         epoch)
        writer.add_image("images/test/actual", make_grid(actual_images_test),
                         epoch)
        writer.add_image("images/test/output", make_grid(output_images_test),
                         epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        if (epoch + 1) % args.save_model_every == 0:
            torch.save(
                net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    torch.save(net, os.path.join(model_save_output, "net_final.p"))


if __name__ == "__main__":
    main()
