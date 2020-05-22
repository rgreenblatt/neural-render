import os
import pickle

import torchvision
import torch
import numpy as np
import OpenEXR as exr
from Imath import PixelType

def load_exr(path):
    file = exr.InputFile(path)

    header = file.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channels = []
    for c in ["R", "G", "B"]:
        channel = file.channel(c, PixelType(PixelType.FLOAT))
        channel = np.fromstring(channel, dtype=np.float32)
        channel = np.reshape(channel, isize)

        channels.append(channel)

    return np.stack(channels, axis=2)

# Note: expects tensor type in standard format (N x C x H x W)
def linear_to_srgb(img):
    return torch.where(img <= 0.0031308, 12.92 * img,
                       1.055 * torch.pow(img, 1 / 2.4) - 0.055)

# Note: expects numpy format image (H x W x C)
# expects square format and clean multiple
def resize(img, output_size):
    input_size = img.shape[0]
    bin_size = input_size // output_size

    assert bin_size * output_size == input_size, "multiple must be exact"

    return img.reshape(
        (output_size, bin_size, output_size, bin_size, 3)).mean(3).mean(1)


class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self, p_path, img_path, resolution, transform=None):
        with open(p_path, "rb") as f:
            self.data = pickle.load(f)
        self.img_path = img_path
        self.transform = transform
        self.resolution = resolution

    def __getitem__(self, index):
        image = load_exr(
            os.path.join(self.img_path,
                         "img_{}.exr".format(index)))

        assert image.shape[0] == image.shape[1], "must be square"

        if image.shape[0] != self.resolution:
            image = resize(image, self.resolution)

        inp = self.data[index]

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


# compare to SubsetRandomSampler
class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def variable_length_collate_fn(batch):
    splits = []
    prev_c = 0
    for b in batch:
        b = b["inp"]
        next_c = prev_c + b.size(0)
        splits.append((prev_c, next_c))

        prev_c = next_c

    inp_cat = torch.cat([b["inp"] for b in batch], dim=0)

    image_stack = torch.stack([b["image"] for b in batch])

    return splits, {"inp": inp_cat, "image": image_stack}

def load_dataset(p_path,
                 img_path,
                 resolution,
                 batch_size,
                 validation_prop,
                 seed,
                 shuffle_split,
                 num_workers=0):
    dataset = RenderedDataset(p_path,
                              img_path,
                              resolution,
                              transform=ToTensor())

    dataset_size = len(dataset)

    np.random.seed(seed)

    indices = list(range(dataset_size))
    split = int(dataset_size * validation_prop)

    if shuffle_split:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = SubsetSampler(val_indices)

    def make_loader(sampler):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=variable_length_collate_fn,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    return train_loader, val_loader
