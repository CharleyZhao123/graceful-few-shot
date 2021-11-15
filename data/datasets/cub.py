import os
import pickle
from PIL import Image

import torch
import clip
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .datasets import dataset_register


@dataset_register('cub')
class CUB(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_file = '{}.csv'.format(split)
        IMAGE_PATH = os.path.join(root_path)
        SPLIT_PATH = os.path.join(root_path, 'split', split_file)

        lines = [x.strip() for x in open(SPLIT_PATH, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        if split == 'train':
            lines.pop(5846)  # this image is broken

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = os.path.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        image_size = 84

        self.data = data
        self.label = label
        self.n_classes = np.unique(np.array(label)).shape[0]

        normalize = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        self.default_transform=transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment=kwargs.get('augment')
        if augment == 'resize':
            self.transform=transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _, preprocess = clip.load('ViT-B/32', device)
            self.transform = preprocess

        else:
            self.transform = self.default_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
