import os

import torch

from model import Net
from load_data import load_dataset

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = 'data/'
    npy_path = os.path.join(data_path, 'transforms.npy')
    img_path = os.path.join(data_path, 'imgs')
    batch_size = 256
    valid_prop = 0.2
    seed = 7

    train, test = load_dataset(npy_path,
                               img_path,
                               batch_size,
                               valid_prop,
                               seed,
                               True,
                               num_workers=8)

    criterion = torch.nn.PairwiseDistance(p=1.0)
    optimizer = torch.optim.Adam


if __name__ == "__main__":
    main()
