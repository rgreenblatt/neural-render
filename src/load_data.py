import os
import math
import pickle
from collections import defaultdict

import torchvision
import torch
import numpy as np
import itertools

from utils import resize
from data_utils import load_exr
from gen_utils import fill_to_size


def chunks_drop_last(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if len(lst[i:]) < n:
            return
        yield lst[i:i + n]


def get_prop_emissive(x):
    count_emissive = sum(map(lambda x: 1 if (x[17:] > 0).all() else 0, x))

    return count_emissive / x.shape[0]


class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self, pickle_path, get_img_path, resolution, transform,
                 fake_data, process_input, data_count_limit):
        if not fake_data:
            with open(pickle_path, "rb") as f:
                self.data = pickle.load(f)

                if data_count_limit is not None:
                    self.data = self.data[:data_count_limit]

                seq_lens = list(map(lambda x: x.shape[0], self.data))
                self.overall_max_seq_len = max(seq_lens)
                self.overall_avg_seq_len = np.mean(seq_lens)

                emissive_props = list(map(get_prop_emissive, self.data))
                self.max_prop_emissive = max(emissive_props)
                self.min_prop_emissive = min(emissive_props)
                self.avg_prop_emissive = np.mean(emissive_props)
        else:
            self.overall_max_seq_len = 1
            self.overall_avg_seq_len = 1
            self.max_prop_emissive = 0.5
            self.min_prop_emissive = 0.5
            self.avg_prop_emissive = 0.5
            self.data_count = (data_count_limit
                               if data_count_limit is not None else 65536)

        self.get_img_path = get_img_path
        self.transform = transform
        self.resolution = resolution
        self.fake_data = fake_data
        self.process_input = process_input

    def __getitem__(self, index):
        if self.fake_data:
            image = np.zeros((self.resolution, self.resolution, 3))
            inp = np.ones((1, 20))  # TODO: fix hardcoded size...
        else:
            image = load_exr(self.get_img_path(index))
            inp = self.data[index]
            inp = fill_to_size(inp, 100)

            assert image.shape[0] == image.shape[1], "must be square"

            if image.shape[0] != self.resolution:
                image = resize(image, self.resolution)

        inp = self.process_input(inp)

        sample = {'image': image, 'inp': inp}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.fake_data:
            return self.data_count
        else:
            return len(self.data)

    def filter_indexes_valid(self, indexes, is_valid):
        if self.fake_data:
            return indexes
        else:
            return list(filter(lambda i: is_valid(self.data[i]), indexes))


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


def mask_collate_fn(samples):
    images = torch.stack([x['image'] for x in samples])

    max_seq_size = max((x['inp'].size(0) for x in samples))

    assert len(samples) > 0

    example_input_sample = samples[0]['inp']

    batch_size = len(samples)
    input_size = example_input_sample.size(1)
    dtype = example_input_sample.dtype
    device = example_input_sample.device

    padded_inputs = torch.empty((batch_size, max_seq_size, input_size),
                                dtype=dtype,
                                device=device)
    masks = torch.empty((batch_size, max_seq_size), dtype=dtype, device=device)
    counts = torch.empty((batch_size, ), dtype=dtype, device=device)

    for i in range(batch_size):
        inp = samples[i]['inp']
        this_seq_size = inp.size(0)
        zero_pad = torch.zeros((max_seq_size - this_seq_size, input_size),
                               dtype=dtype,
                               device=device)
        padded_inputs[i] = torch.cat((inp, zero_pad))

        ones_mask = torch.ones((this_seq_size, ), dtype=dtype, device=device)
        zeros_mask = torch.zeros((max_seq_size - this_seq_size, ),
                                 dtype=dtype,
                                 device=device)
        masks[i] = torch.cat((ones_mask, zeros_mask))
        counts[i] = this_seq_size

    return {
        'image': images,
        'inp': padded_inputs,
        'mask': masks,
        'count': counts,
    }


class SubsetRandomDistributedSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without
    replacement ensuring groups remain together.

    Arguments:
        subset (sequence): a sequence of indices
        group_size (int): the size to keep together
        shuffle (bool): whether or not to random shuffle
        TODO
    """
    def __init__(self, subset, num_replicas=1, rank=0, shuffle=True):
        self.subset = subset
        self.shuffle = shuffle

        self.epoch = 0

        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = math.ceil(len(self.subset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.subset), generator=g).tolist()
        else:
            indices = list(range(len(self.subset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return (self.subset[i] for i in indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DatasetManager():
    def __init__(self,
                 pickle_path,
                 get_img_path,
                 resolution,
                 batch_size,
                 validation_prop,
                 max_validation_size,
                 seed,
                 num_workers=0,
                 fake_data=False,
                 process_input=lambda x: x,
                 data_count_limit=None,
                 num_replicas=1,
                 rank=0):
        self.dataset = RenderedDataset(pickle_path,
                                       get_img_path,
                                       resolution,
                                       transform=ToTensor(),
                                       fake_data=fake_data,
                                       process_input=process_input,
                                       data_count_limit=data_count_limit)
        self.overall_max_seq_len = self.dataset.overall_max_seq_len
        self.overall_avg_seq_len = self.dataset.overall_avg_seq_len
        self.max_prop_emissive = self.dataset.max_prop_emissive
        self.min_prop_emissive = self.dataset.min_prop_emissive
        self.avg_prop_emissive = self.dataset.avg_prop_emissive

        num_samples = len(self.dataset)

        np.random.seed(seed)

        indices = list(range(num_samples))
        split = min(int(num_samples * validation_prop), max_validation_size)

        np.random.shuffle(indices)
        self.train_subset, self.val_subset = indices[split:], indices[:split]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.rank = rank

    def get_train_test(self, max_seq_len, min_prop_emissive):
        def is_valid(x):
            seq_size = max_seq_len is None or x.shape[0] <= max_seq_len

            prop_emissive = get_prop_emissive(x)

            emissive = (min_prop_emissive is None
                        or prop_emissive >= min_prop_emissive)

            return seq_size and emissive

        train_subset = self.dataset.filter_indexes_valid(
            self.train_subset, is_valid)
        val_subset = self.dataset.filter_indexes_valid(self.val_subset,
                                                       is_valid)

        train_sampler = SubsetRandomDistributedSampler(
            train_subset, num_replicas=self.num_replicas, rank=self.rank)
        val_sampler = SubsetRandomDistributedSampler(
            val_subset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=False)

        def make_loader(sampler):
            return torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
                collate_fn=mask_collate_fn,
            )

        train_loader = make_loader(train_sampler)
        val_loader = make_loader(val_sampler)

        def epoch_callback(epoch):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)  # not really required

        return train_loader, val_loader, epoch_callback
