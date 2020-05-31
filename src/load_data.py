import os
import math
import pickle
from collections import defaultdict

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

def chunks_drop_last(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if len(lst[i:]) < n:
            return
        yield lst[i:i + n]

class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, img_path, resolution, transform, fake_data,
                 process_input, start_range, end_range):
        if not fake_data:
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
            inp = np.ones((1, 20)) # TODO: fix hardcoded size...
        else:
            image = load_exr(
                os.path.join(self.img_path, "img_{}.exr".format(index)))

            assert image.shape[0] == image.shape[1], "must be square"

            if image.shape[0] != self.resolution:
                image = resize(image, self.resolution)

            inp = self.data[index]

        inp = self.process_input(inp)

        sample = {'image': image, 'inp': inp}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def groups(self, batch_size):
        grouped = defaultdict(list)

        if self.fake_data:
            grouped = {0: [], 1: [i for i in range(self.fake_data_size)]}
        else:
            for i, inp in enumerate(self.data):
                grouped[inp.shape[0]].append(i)

        out = []

        for group in grouped.values():
            out.extend(chunks_drop_last(group, batch_size))

        return out




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
                 groups,
                 num_replicas=1,
                 rank=0,
                 shuffle=True):
        self.indices = indices
        self.groups = groups
        assert len(self.groups) != 0
        self.group_size = len(self.groups[0])
        self.shuffle = shuffle

        self.epoch = 0

        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

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

        return (self.groups[self.indices[i]][place]
                for (i, place) in index_and_place_iter)

    def __len__(self):
        return len(self.indices) * self.group_size

    def set_epoch(self, epoch):
        self.epoch = epoch


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

    groups = dataset.groups(batch_size)

    # we load in chunks to ensure each batch has consistant size
    # the dataset must have a consistant size per each group of batch_size
    num_groups = len(groups)

    np.random.seed(seed)

    indices = list(range(num_groups))
    split = int(num_groups * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetGroupedRandomDistributedSampler(
        train_indices, groups, num_replicas=num_replicas, rank=rank)
    val_sampler = SubsetGroupedRandomDistributedSampler(
        val_indices,
        groups,
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

    def epoch_callback(epoch):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)  # not really required

    return train_loader, val_loader, epoch_callback
