import os
import pickle
import random

import torch
from torch.utils.data import Dataset
import numpy as np

from .datasets import model_register

default_split = {
    'train': 0.7,
    'val': 0.3,
}


@model_register('gb-100')
class GB100(Dataset):
    def __init__(self, root_path, split='train', split_method='novel', **kwargs):
        data_file_name = 'gb_dataset.pickle'
        with open(os.path.join(root_path, data_file_name), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')

        # 经过默认数据处理[Resize, ToTensor, normalize]的图像tensor，可直接输入Network
        default_data = pack['data']
        feature = pack['feature']
        imgname = pack['imgname']
        origin_label = pack['origin_label']
        logits = pack['logits']
        gb_label = pack['gb_label']

        # 划分数据
        g_index = []
        b_index = []
        for i, l in enumerate(gb_label):
            if l == 1.0:
                g_index.append(i)
            else:
                b_index.append(i)

        if split_method == 'random':
            # 随机抽取数据并划分数据
            random.seed(0)
            train_g_index = random.sample(g_index, int(
                len(g_index)*default_split['train']))
            val_g_index = list(set(g_index).difference(set(train_g_index)))

            random.seed(1)
            train_b_index = random.sample(b_index, int(
                len(b_index)*default_split['train']))
            val_b_index = list(set(b_index).difference(set(train_b_index)))

            train_index = train_g_index + train_b_index
            val_index = val_g_index + val_b_index
        else:
            # 前n个class为训练集, 后64-n个为验证集划分数据
            t_class_num =  int(default_split['train'] * 64)  # n
            v_class_num = 64 - t_class_num

            train_g_index = g_index[:100*t_class_num]
            val_g_index = g_index[100*t_class_num:]

            train_b_index = b_index[:100*t_class_num]
            val_b_index = b_index[100*t_class_num:]

            train_index = train_g_index + train_b_index
            val_index = val_g_index + val_b_index

        if split == 'train':
            self.index_list = train_index
        else:
            self.index_list = val_index

        self.data = default_data
        self.feature = feature
        self.gb_label = gb_label

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        index = self.index_list[i]
        return self.data[index], self.feature[index], int(self.gb_label[index])


if __name__ == '__main__':
    gb_100 = GB100(
        root_path='/space1/zhaoqing/dataset/fsl/gb-100', split='val', split_method='novel')
    print(len(gb_100))

    # random
    # val 3840
    # train 8960
    # novel
    # val 4000
    # train 8800
