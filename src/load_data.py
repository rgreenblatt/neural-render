import os

import torchvision
import torch
import numpy as np
from skimage import io


class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, img_path, transform=None):
        self.data = np.load(npy_path)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        image = io.imread(os.path.join(self.img_path,
                                       "img_{}.png".format(index)))
        label = self.data[index]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)


        return sample

    def __len__(self):
        return len(self.data)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

def load_dataset(npy_path,
                 img_path,
                 batch_size,
                 validation_prop,
                 seed,
                 shuffle_split,
                 num_workers=0):
    dataset = RenderedDataset(
        npy_path, img_path, transform=ToTensor())

    dataset_size = len(dataset)

    np.random.seed(seed)

    indices = list(range(dataset_size))
    split = int(dataset_size * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    def make_loader(sampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    return train_loader, val_loader
