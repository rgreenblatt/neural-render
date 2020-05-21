import os
import argparse

import torch
import numpy as np
import pytorch_model_summary as pms
from torchvision.utils import save_image

from model import Net
from load_data import load_dataset
from arch import net_params
from utils import mkdirs, PiecewiseLinear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-multiplier', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=128)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = 'data/'
    npy_path = os.path.join(data_path, 'transforms.npy')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = args.batch_size
    valid_prop = 0.2
    seed = 7

    img_width = args.resolution

    train, test = load_dataset(npy_path,
                               img_path,
                               batch_size,
                               valid_prop,
                               seed,
                               True,
                               num_workers=8)

    blocks_args, global_params = net_params(chan_multiplier=1.5,
                                            base_min_ch=32,
                                            output_width=img_width)

    net = Net(blocks_args, global_params).to(device)

    print(pms.summary(net, torch.zeros(1, 3, device=device).float()))
    print(net)

    # TODO: consider switching to l1 (as opposed to l2)
    criterion = torch.nn.MSELoss()
    epoches = 100
    lr_schedule = PiecewiseLinear([0, 10, 70, 100],
                                  [0.001, 0.005, 0.0005, 0.00005])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)

    mkdirs("outputs")

    for epoch in range(epoches):
        net.train()
        train_loss = 0.0

        lr = args.lr_multiplier * lr_schedule(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, data in enumerate(train):
            inp = data['inp'].to(device)
            image = data['image'].to(device)

            optimizer.zero_grad()

            outputs = net(inp)

            zeros = torch.zeros_like(image)

            if i % 100 == 0:
                save_image(image,
                           "outputs/train_actual_{}_{}.png".format(epoch, i))
                save_image(outputs,
                           "outputs/train_output_{}_{}.png".format(epoch, i))

            loss = criterion(outputs, image)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train)

        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test):
                inp = data['inp'].to(device)
                image = data['image'].to(device)

                outputs = net(inp)

                if i % 30 == 0:
                    save_image(
                        image,
                        "outputs/test_actual_{}_{}.png".format(epoch, i))
                    save_image(
                        outputs,
                        "outputs/test_output_{}_{}.png".format(epoch, i))

                loss = criterion(outputs, image)

                test_loss += loss.item()

        test_loss /= len(test)

        print("epoch: {}, train loss: {}, test loss: {}, lr: {}".format(
            epoch, train_loss, test_loss, lr))

    example_transforms = np.array([[4.5, 4.2, 4.5], [-4.5, -4.2, -4.5],
                                   [-1.0, 0.0, 0.0]])

    def evaluate_and_save_examples(transforms):
        with torch.no_grad():
            count = transforms.shape[0]
            transforms = torch.tensor(transforms).float().to(device)

            outputs = net(transforms).detach().cpu()
            for i in range(count):
                save_image(outputs[i] * 255,
                           "outputs/example_img_{}.png".format(i))

    evaluate_and_save_examples(example_transforms)


if __name__ == "__main__":
    main()
