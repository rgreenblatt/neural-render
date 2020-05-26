import os
import pickle

import torchvision
import torch
import numpy as np
import imageio
from utils import resize


def load_exr(path):
    return imageio.imread(path)[:,:,:3]

def write_exr(path, img):
    return imageio.imwrite(path, img)

class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 p_path,
                 img_path,
                 resolution,
                 transform,
                 fake_data,
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

        inp = process_input(self.data[index])

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
                 num_workers=0,
                 fake_data=False,
                 process_input=lambda x: x):
    dataset = RenderedDataset(p_path,
                              img_path,
                              resolution,
                              transform=ToTensor(),
                              fake_data=fake_data,
                              process_input=process_input)

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
            pin_memory=True,
        )

    train_loader = make_loader(train_sampler)
    val_loader = make_loader(val_sampler)

    return train_loader, val_loader
