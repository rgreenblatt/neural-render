import os
import math
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
                 process_input, start_range, end_range):
        with open(p_path, "rb") as f:
            self.data = pickle.load(f)[start_range:end_range]
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

class SubsetGroupedRandomDistributedSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without
    replacement ensuring groups remain together.

    Arguments:
        indices (sequence): a sequence of indices
        group_size (int): the size to keep together
        shuffle (bool): whether or not to random shuffle
        TODO
    """
    def __init__(self,
                 indices,
                 group_size,
                 num_replicas=1,
                 rank=0,
                 shuffle=True):
        self.indices = indices
        self.group_size = group_size
        self.shuffle = shuffle

        self.epoch = 0

        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()

        if self.num_replicas != 1:
            g.manual_seed(self.epoch)
        else:
            g.manual_seed(torch.random.seed())

        if self.shuffle:
            indices = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            indices = list(range(len(self.indices)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        index_and_place_iter = itertools.product(indices,
                                                 range(self.group_size))

        return (self.indices[i] * self.group_size + place
                for (i, place) in index_and_place_iter)

    def __len__(self):
        return self.num_samples * self.group_size


def load_dataset(p_path,
                 img_path,
                 resolution,
                 batch_size,
                 validation_prop,
                 seed,
                 shuffle_split,
                 num_workers=0,
                 fake_data=False,
                 process_input=lambda x: x,
                 start_range=0,
                 end_range=-1,
                 num_replicas=1,
                 rank=0):
    dataset = RenderedDataset(p_path,
                              img_path,
                              resolution,
                              transform=ToTensor(),
                              fake_data=fake_data,
                              process_input=process_input,
                              start_range=start_range,
                              end_range=end_range)

    # we load in chunks to ensure each batch has consistant size
    # the dataset must have a consistant size per each group of batch_size
    dataset_size = len(dataset) // batch_size

    np.random.seed(seed)

    indices = list(range(dataset_size))
    split = int(dataset_size * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetGroupedRandomDistributedSampler(
        train_indices, batch_size, num_replicas=num_replicas, rank=rank)
    val_sampler = SubsetGroupedRandomDistributedSampler(
        val_indices,
        batch_size,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=False)

    def make_loader(sampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    return train_loader, val_loader
