import os
import pickle
from PIL import Image
import random

import torch
import clip
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import dataset_register


@dataset_register('img-mini-imagenet')
class ImgMiniImageNet(Dataset):

    def __init__(self, root_path, split='train', patch_type='none', **kwargs):
        split_file = '{}.csv'.format(split)
        IMAGE_PATH = os.path.join(root_path, 'images')
        SPLIT_PATH = os.path.join(root_path, 'split', split_file)

        self.patch_type = patch_type  # 划分Patch的方式

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

        # ===== 数据增强设置 =====
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)

        # 非Patch模式下的数据增强
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
        elif augment == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _, preprocess = clip.load('ViT-B/32', device)
            self.transform = preprocess

        else:
            self.transform = self.default_transform

        # Patch模式下的数据增强
        if self.patch_type == 'sampling':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                # transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                #                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
            if augment == 'clip':
                image_size = 224
                self.default_transform = preprocess
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    self.default_transform
                ])

        if self.patch_type == 'grid':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            if augment == 'clip':
                image_size = 224
                self.default_transform = preprocess
                self.transform = transforms.Compose([
                    transforms.Resize([image_size, image_size]),
                    transforms.RandomHorizontalFlip(),
                    self.default_transform
                ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def get_grid_location(self, size, ratio, num_grid):
        '''

        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_patch):

        num_grid = num_patch
        grid_ratio = 1+2*random.random()

        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop(
                    (left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform(patch)
                patches_list.append(patch)

        return patches_list

    def __getitem__(self, i):
        imgname = self.data[i][-21:]
        if self.patch_type == 'none':
            img = Image.open(self.data[i]).convert('RGB')
            return self.transform(img), self.label[i], imgname
        elif self.patch_type == 'gird':
            patch_list = []
            patch_nums = [3, 2]  # 一共3x3+2x2=13个patch

            image = Image.open(self.data[i]).convert('RGB')

            # patch_list第一个元素为原图
            patch_list.append(self.default_transform(image))

            # patch_list后续元素为patch图
            for num_patch in patch_nums:
                patches = self.get_pyramid(image, num_patch)
                patch_list.extend(patches)

            # [1+patch_num, 3, 80, 80]
            patch_list = torch.stack(patch_list, dim=0)
            return patch_list, self.label[i], imgname

        elif self.patch_type == 'sampling':
            patch_list = []

            # patch_list第一个元素为原图
            patch_list.append(self.default_transform(
                Image.open(self.data[i]).convert('RGB')))

            # patch_list后续元素为patch图
            extra_patch_num = 19
            for _ in range(extra_patch_num):
                patch_list.append(self.transform(
                    Image.open(self.data[i]).convert('RGB')))

            # [1+patch_num, 3, 80, 80]
            patch_list = torch.stack(patch_list, dim=0)
            return patch_list, self.label[i], imgname
