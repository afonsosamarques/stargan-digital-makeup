import torch
from torch.utils import data
from torchvision import transforms as tr

from PIL import Image

import os
import random
from datetime import datetime


class NoMakeup(data.Dataset):
    def __init__(self, image_dir, filename_path, transform, mode, test_set=0):
        self.image_dir = image_dir
        self.filename_path = filename_path
        self.transform = transform
        self.mode = mode
        self.test_set = test_set

        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        # Removing all end of line characters and consider lines corresponding to labels
        lines = [line.rstrip() for line in open(self.filename_path, 'r')]
        lines = lines[1:]

        # Mixing it up
        random.seed(datetime.now())
        random.shuffle(lines)
        for i, line in enumerate(lines):
            line_array = line.split()
            filename = line_array[0]
            if i < self.test_set:
                self.test_dataset.append(filename)
            else:
                self.train_dataset.append(filename)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image)

    def __len__(self):
        return self.num_images


class Makeup(data.Dataset):
    def __init__(self, image_dir, att_path, transform, mode):
        self.image_dir = image_dir
        self.att_path = att_path
        self.transform = transform
        self.mode = mode

        self.dataset = []
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        # Removing all end of line characters and consider lines corresponding to labels
        lines = [line.rstrip() for line in open(self.att_path, 'r')]
        lines = lines[1:]

        # Mixing it up
        random.seed(datetime.now())
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            label = split[1:]
            self.dataset.append([filename, label])

    def __getitem__(self, index):
        filename, label = self.dataset[index]
        label = list(map(int, label))
        label = torch.Tensor(label)
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), label

    def __len__(self):
        return self.num_images


def get_loader(image_dir, crop_size, image_size, dataset, att_path, batch_size=16, mode='train', test_set=None, num_workers=1):
    transform_general = []
    transform_general.append(tr.Resize(image_size, interpolation=Image.BICUBIC))
    transform_general.append(tr.CenterCrop(crop_size))
    transform_general.append(tr.ToTensor())
    transform_general.append(tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform_general = tr.Compose(transform_general)

    if mode == 'train':
        transform_train = []
        transform_train.append(tr.RandomHorizontalFlip(p=1.0))
        transform_train.append(tr.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0))
        transform_train.append(tr.Resize(image_size, interpolation=Image.BICUBIC))
        transform_train.append(tr.CenterCrop(crop_size))
        transform_train.append(tr.ToTensor())
        transform_train.append(tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform_train = tr.Compose(transform_train)

    # Setup datasets
    if dataset == 'NoMakeup':
        dataset = NoMakeup(image_dir, att_path, transform_general, mode, test_set=test_set)
        if mode == 'train':
            extra_data = NoMakeup(image_dir, att_path, transform_train, mode, test_set=test_set)
            dataset = data.ConcatDataset([dataset, extra_data])
    elif dataset == 'Makeup':
        dataset = Makeup(image_dir, att_path, transform_general, mode)
        if mode == 'train':
            extra_data = Makeup(image_dir, att_path, transform_train, mode)
            dataset = data.ConcatDataset([dataset, extra_data])

    # Build data loader
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers
    )

    return data_loader
