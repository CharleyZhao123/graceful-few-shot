import os
import pickle
from PIL import Image
import json

import torch
import clip
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import dataset_register

default_split = {
    'train': 0.7,
    'val': 0.3,
}

name2label = {
    'person': 0,
    'tank': 3,
    'carrier': 2,
    'armored': 1,
    'car': 4,
    'radar': 5,
    'launch': 6
}


@dataset_register('mushi')
class Mushi(Dataset):
    def __init__(self, root_path, split='train', augment='default',
                 type='sim_data', shot_num=70, query_num=15, return_items=2, **kwargs):

        self.root_path = root_path
        self.split = split
        self.type = type

        # ===== 按照数据集类型加载和整理数据 =====
        if self.type == 'sim_data':
            self.image, self.label = self.get_sim_data(shot_num, query_num)
        elif self.type == 'true_data':
            self.image, self.label = self.get_true_data(shot_num, query_num)
        elif self.type == 'mix_data':
            self.image, self.label = self.get_mix_data(shot_num, query_num)

        # ===== 预处理数据 =====
        image_size = 224  # 80
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])
        if augment == 'crop':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'resize':
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),  # transforms.Resize(image_size)
                # transforms.RandomCrop(image_size, padding=8),
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

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

        # 其他
        self.return_items = return_items

    def get_sim_data(self, shot_num, query_num, sim_type='mocod'):
        '''
        训练集: 仿真数据
        测试集: 仿真数据
        '''
        # 加载仿真数据集信息json文件
        if sim_type == 'mocod':
            json_file = 'dataset_info.json'
        else:
            json_file = 'gan_mushi_info.json'

        info_json_file = os.path.join(self.root_path, json_file)
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
            f_train_image = f['images'][:f_train_num][:shot_num]
            f_train_label = ([f_label] * f_train_num)[:shot_num]

            train_image += f_train_image
            train_label += f_train_label

            # 划分给验证集
            f_val_num = f_class_num - f_train_num
            f_val_image = f['images'][f_train_num:][:query_num]
            f_val_label = ([f_label] * f_val_num)[:query_num]

            val_image += f_val_image
            val_label += f_val_label

        if self.split == 'train':
            image = train_image
            label = train_label
        else:
            image = val_image
            label = val_label

        return image, label

    def get_true_data(self, shot_num, query_num):
        '''
        训练集: 真实数据
        测试集: 真实数据
        '''
        true_floder_path = os.path.join(self.root_path, 'true')
        class_floder_list = os.listdir(true_floder_path)

        # 整理数据
        train_image = []
        train_label = []

        val_image = []
        val_label = []
        for c in class_floder_list:
            c_label = int(name2label[c])

            c_floder_path = os.path.join(true_floder_path, c)
            c_image_list = os.listdir(c_floder_path)

            c_image_list.sort(key=lambda x: int(x[:-4]))
            # print(c_image_list)
            # 处理image path
            c_image_list = [os.path.join('true', c, p) for p in c_image_list]

            c_class_num = len(c_image_list)

            # 划分给训练集
            c_train_num = int(c_class_num*default_split['train'])
            c_train_image = c_image_list[:c_train_num][:shot_num]
            c_train_label = ([c_label] * c_train_num)[:shot_num]

            train_image += c_train_image
            train_label += c_train_label

            # 划分给验证集
            c_val_num = c_class_num - c_train_num
            c_val_image = c_image_list[c_train_num:][-query_num:]
            c_val_label = ([c_label] * c_val_num)[-query_num:]

            val_image += c_val_image
            val_label += c_val_label

        if self.split == 'train':
            image = train_image
            label = train_label
        else:
            image = val_image
            label = val_label

        return image, label

    def get_mix_data(self, shot_num, query_num):
        '''
        训练集: 仿真和真实数据按比例混合得到
        测试集: 全部为真实数据
        '''
        # 训练集为混合数据
        if self.split == 'train':
            sim_true_rate = {
                'sim': 0.7,
                'mocod': 0.5,
                'gan': 0.5,
                'true': 0.3
            }
            sim_shot_num = int(shot_num*sim_true_rate['sim'])
            true_shot_num = shot_num - sim_shot_num
            # 仿真数据
            mocod_shot_num = int(sim_shot_num*sim_true_rate['mocod'])
            gan_shot_num = sim_shot_num - mocod_shot_num

            mocod_image, mocod_label = self.get_sim_data(mocod_shot_num, query_num, 'mocod')
            gan_image, gan_label = self.get_sim_data(gan_shot_num, query_num, 'gan')

            sim_image = mocod_image + gan_image
            sim_label = mocod_label + gan_label
            # 真实数据
            true_image, true_label = self.get_true_data(
                true_shot_num, query_num)

            image = sim_image + true_image
            label = sim_label + true_label
        # 测试集为真实数据
        else:
            image, label = self.get_true_data(shot_num, query_num)

        return image, label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index].replace(
            'label', 'origin').replace('json', 'png')
        image_path = os.path.join(self.root_path, image_path)
        # print(image_path)
        image = Image.open(image_path).convert('RGB')

        if self.return_items == 3:
            fake_data = 'fake_name'
            return self.transform(image), self.label[index], fake_data
        else:
            return self.transform(image), self.label[index]


if __name__ == '__main__':
    mushi = Mushi(
        root_path='/space1/zhaoqing/dataset/fsl/mushi', split='val', type='true_data')
    print(mushi.__getitem__(0))
    # sim: train: 402, val: 178;
    # true v1: train: 407 val: 174;
    # true v2: train: 403 val: 171;
    print(len(mushi))
