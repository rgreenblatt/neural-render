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
    def __init__(self, pickle_path, get_img_path, resolution, transform, fake_data,
                 process_input, start_range, end_range):
        if not fake_data:
            with open(pickle_path, "rb") as f:
                self.data = pickle.load(f)[start_range:end_range]
        self.get_img_path = get_img_path
        self.transform = transform
        self.resolution = resolution
        self.fake_data = fake_data
        self.process_input = process_input

    def __getitem__(self, index):
        if self.fake_data:
            image = np.zeros((self.resolution, self.resolution, 3))
            inp = np.ones((1, 20)) # TODO: fix hardcoded size...
        else:
            image = load_exr(get_img_path(index))

            assert image.shape[0] == image.shape[1], "must be square"

            if image.shape[0] != self.resolution:
                image = resize(image, self.resolution)

            inp = self.data[index]

        inp = self.process_input(inp)

        sample = {'image': image, 'inp': inp}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        if self.fake_data:
            return 65536
        else:
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
        zero_pad = torch.zeros(
            (max_seq_size - this_seq_size, input_size),
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
    def __init__(self,
                 subset,
                 num_replicas=1,
                 rank=0,
                 shuffle=True):
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


def load_dataset(pickle_path,
                 get_img_path,
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
    dataset = RenderedDataset(pickle_path,
                              get_img_path,
                              resolution,
                              transform=ToTensor(),
                              fake_data=fake_data,
                              process_input=process_input,
                              start_range=start_range,
                              end_range=end_range)

    # we load in chunks to ensure each batch has consistant size
    # the dataset must have a consistant size per each group of batch_size
    num_samples = len(dataset)

    np.random.seed(seed)

    indices = list(range(num_samples))
    split = int(num_samples * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_subset, val_subset = indices[split:], indices[:split]

    train_sampler = SubsetRandomDistributedSampler(train_subset,
                                                   num_replicas=num_replicas,
                                                   rank=rank)
    val_sampler = SubsetRandomDistributedSampler(val_subset,
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
            collate_fn=mask_collate_fn,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    def epoch_callback(epoch):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)  # not really required

    return train_loader, val_loader, epoch_callback
