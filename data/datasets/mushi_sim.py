import os
import pickle
from PIL import Image
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

default_split = {
    'train': 0.7,
    'val': 0.3,
}

name2label = {
    'person': 0,
    'tank': 1,
    'carrier': 2,
    'armored': 3,
    'car': 4,
    'radar': 5,
    'launch': 6
}


@register('mushi-sim')
class MushiSIM(Dataset):
    def __init__(self, root_path, split='train', augment='default', **kwargs):

        # ===== 加载和整理数据 =====
        # 加载数据集信息json文件
        self.root_path = root_path
        info_json_file = os.path.join(root_path, 'dataset_info.json')
        with open(info_json_file, 'r') as json_f:
            info_dict = json.load(json_f)

        # 整理数据
        train_image = []
        train_label = []

        val_image = []
        val_label = []

        father_info = info_dict['fathers']
        for f in father_info:
            f_class_num = f['num']
            f_label = int(name2label[f['name']])

            # 暂时跳过radar和launch
            if f_label == 5 or f_label == 6:
                continue

            # 划分给训练集
            f_train_num = int(f_class_num*default_split['train'])
            f_train_image = f['images'][:f_train_num]
            f_train_label = [f_label] * f_train_num

            train_image += f_train_image
            train_label += f_train_label

            # 划分给验证集
            f_val_num = f_class_num - f_train_num
            f_val_image = f['images'][f_train_num:]
            f_val_label = [f_label] * f_val_num

            val_image += f_val_image
            val_label += f_val_label

        if split == 'train':
            self.image = train_image
            self.label = train_label
        else:
            self.image = val_image
            self.label = val_label

        # ===== 预处理数据 =====
        image_size = 80
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
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index].replace(
            'label', 'origin').replace('json', 'png')
        image_path = os.path.join(self.root_path, image_path)
        # print(image_path)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.label[index]


if __name__ == '__main__':
    mushi_sim = MushiSIM(
        root_path='/space1/zhaoqing/dataset/fsl/mushi-sim', split='val')
    print(mushi_sim.__getitem__(0))
    print(len(mushi_sim))  # train: 402, val: 178
