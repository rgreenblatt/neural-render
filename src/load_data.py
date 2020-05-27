import os
import pickle

import torchvision
import torch
import numpy as np
import imageio
from utils import resize
import itertools


def load_exr(path):
    return imageio.imread(path)[:, :, :3]


def write_exr(path, img):
    return imageio.imwrite(path, img)


class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, img_path, resolution, transform, fake_data,
                 process_input):
        with open(p_path, "rb") as f:
            self.data = pickle.load(f)
        self.img_path = img_path
        self.transform = transform
        self.resolution = resolution
        self.fake_data = fake_data
        self.process_input = process_input

    def __getitem__(self, index):
        if self.fake_data:
            image = np.zeros((self.resolution, self.resolution, 3))
        else:
            image = load_exr(
                os.path.join(self.img_path, "img_{}.exr".format(index)))

            assert image.shape[0] == image.shape[1], "must be square"

            if image.shape[0] != self.resolution:
                image = resize(image, self.resolution)

        inp = self.process_input(self.data[index])

        sample = {'image': image, 'inp': inp}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, inp = sample['image'], sample['inp']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image).float(),
            'inp': torch.from_numpy(inp).float()
        }


def get_grouped_index(indexes, group_size, i):
    return indexes[i // group_size] * group_size + i % group_size


# compare to SubsetRandomSampler
class SubsetGroupedSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements from a given list of indices, without replacement
    ensuring groups remain together.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices, group_size):
        self.indices = indices
        self.group_size = group_size

    def __iter__(self):
        return (get_grouped_index
                for i in range(len(self.indices) * self.group_size))

    def __len__(self):
        return len(self.indices) * self.group_size


class SubsetGroupedRandomSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without
    replacement ensuring groups remain together.

    Arguments:
        indices (sequence): a sequence of indices
        group_size (int): the size to keep together
        rand_perm (bool): whether or not to random shuffle
    """
    def __init__(self, indices, group_size, rand_perm=True):
        self.indices = indices
        self.group_size = group_size
        self.rand_perm = rand_perm

    def __iter__(self):
        length = len(self.indices)
        index_iter = torch.randperm(length) if self.rand_perm else range(
            length)
        index_and_place_iter = itertools.product(index_iter,
                                                 range(self.group_size))
        return (self.indices[i] * self.group_size + place
                for (i, place) in index_and_place_iter)

    def __len__(self):
        return len(self.indices) * self.group_size


def load_dataset(p_path,
                 img_path,
                 resolution,
                 batch_size,
                 validation_prop,
                 seed,
                 shuffle_split,
                 num_workers=0,
                 fake_data=False,
                 process_input=lambda x: x):
    dataset = RenderedDataset(p_path,
                              img_path,
                              resolution,
                              transform=ToTensor(),
                              fake_data=fake_data,
                              process_input=process_input)

    # we load in chunks to ensure each batch has consistant size
    # the dataset must have a consistant size per each group of batch_size
    dataset_size = len(dataset) // batch_size

    np.random.seed(seed)

    indices = list(range(dataset_size))
    split = int(dataset_size * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetGroupedRandomSampler(train_indices, batch_size)
    val_sampler = SubsetGroupedRandomSampler(val_indices,
                                             batch_size,
                                             rand_perm=False)

    def make_loader(sampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    return train_loader, val_loader
