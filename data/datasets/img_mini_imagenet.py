import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('img-mini-imagenet')
class ImgMiniImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_file = '{}.csv'.format(split)
        IMAGE_PATH = os.path.join(root_path, 'images')
        SPLIT_PATH = os.path.join(root_path, 'split', split_file)

        lines = [x.strip() for x in open(SPLIT_PATH, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        image_size = 80

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize([80, 80]),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
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
        else:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        imgname = self.data[i][-21:]
        img = Image.open(self.data[i]).convert('RGB')
        return self.transform(img), self.label[i], imgname
