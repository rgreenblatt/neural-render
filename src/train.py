import os

import torch
import numpy as np
import pytorch_model_summary as pms
from torchvision.utils import save_image

from model import Net
from load_data import load_dataset
from arch import net_params
from utils import mkdirs, PiecewiseLinear


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = 'data/'
    npy_path = os.path.join(data_path, 'transforms.npy')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = 16
    valid_prop = 0.2
    seed = 7

    img_width = 128

    train, test = load_dataset(npy_path,
                               img_path,
                               batch_size,
                               valid_prop,
                               seed,
                               True,
                               img_width=img_width,
                               num_workers=8)

    blocks_args, global_params = net_params(output_width=img_width)

    net = Net(blocks_args, global_params).to(device)

    print(pms.summary(net, torch.zeros(1, 3, device=device).float()))

    # TODO: consider switching to l1 (as opposed to l2)
    criterion = torch.nn.MSELoss(reduction='sum')
    epoches = 100
    lr_schedule = PiecewiseLinear([0, 20, 80],
                                  [0.1, 3, 0.003])
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.1,
        weight_decay=0.0)

    mkdirs("outputs")

    for epoch in range(epoches):
        net.train()
        train_loss = 0.0

        lr = lr_schedule(epoch) / batch_size

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, data in enumerate(train):
            inp = data['inp'].to(device)
            image = data['image'].to(device) / 255.0

            optimizer.zero_grad()

            outputs = net(inp)

            if i % 100 == 0:
                save_image(image * 255,
                           "outputs/train_actual_{}_{}.png".format(epoch, i))
                save_image(outputs * 255,
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
                image = data['image'].to(device) / 255.0

                outputs = net(inp)

                if i % 30 == 0:
                    save_image(image * 255,
                               "outputs/test_actual_{}_{}.png".format(epoch, i))
                    save_image(outputs * 255,
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
