import torch
import torch.nn as nn
import os
import yaml
import json
import pickle
import utils
import argparse
from data import build_dataloader
from models import build_model
from tqdm import tqdm

base_proto_f_path = 'visualization/saved_data/base_prototype_feature.pth'
base_proto_l_path = 'visualization/saved_data/base_prototype_label.pth'


def get_best_base_sample(config):
    '''
    获得数据集中与Good Prototype距离最近以及最远的图像
    '''

    # ===== 准备数据、模型 =====

    # nn classifier base good prototype
    # base_proto_f = torch.load(base_proto_f_path)
    # base_proto_l = torch.load(base_proto_l_path)

    # dataloader
    train_dataloader = build_dataloader(config['train_dataloader_args'])

    # model
    model = build_model(config['network_args'])
    if config.get('_parallel'):
        model = nn.DataParallel(model)
    model.eval()

    # ===== 记录训练集图像信息 =====

    # 记录或读取类别编号和名称匹配信息
    # classname_dict = {}

    # 记录label, imgname, data-proto-distance信息
    imgname_list = []
    label_tensor = torch.zeros([1])
    data_tensor = torch.zeros([1, 3, 80, 80]).cuda()
    feature_tensor = torch.zeros([1, 512]).cuda()
    logits_tensor = torch.zeros([1, 64]).cuda()

    for data, label, imgname in tqdm(train_dataloader, leave=False):
        data = data.cuda()  # [1, 3, 80, 80]
        if False:
            # 记录类别编号和名称匹配信息
            if imgname[0][0:9] not in classname_dict:
                classname_dict[imgname[0][0:9]] = label[0].item()
        with torch.no_grad():
            feature, logits = model(data)

        imgname_list.append(imgname[0])
        data_tensor = torch.cat((data_tensor, data), 0)
        label_tensor = torch.cat((label_tensor, label), 0)
        feature_tensor = torch.cat((feature_tensor, feature), 0)
        logits_tensor = torch.cat((logits_tensor, logits), 0)

    data_tensor = data_tensor[1:, :, :, :].cpu()  # [38400, 3, 80, 80]
    label_tensor = label_tensor[1:]  # [38400]
    feature_tensor = feature_tensor[1:, :].cpu()  # [38400, 512]
    logits_tensor = logits_tensor[1:, :].cpu()  # [38400, 64]

    # 得到供训练二分类器的pickle文件
    # 64类, 每类找到最好的100张图像, 最差的100张图像;
    #
    gb_dataset = {
        'data': [],
        'feature': [],
        'imgname': [],
        'origin_label': [],
        'logits': [],
        'gb_label': []
    }

    for class_i in range(64):
        class_i = int(class_i)
        class_i_best = {}
        class_i_worst = {}
        class_i_logits = logits_tensor[:, class_i]

        # 寻找最好的100个图像
        b_sorted_class_i_logits, b_index = torch.sort(
            class_i_logits, descending=True)
        for n in range(100):
            n_index = b_index[n]
            gb_dataset['data'].append(data_tensor[n_index, :, :, :].clone())
            gb_dataset['feature'].append(feature_tensor[n_index, :].clone())
            gb_dataset['imgname'].append(imgname_list[n_index])
            gb_dataset['origin_label'].append(label_tensor[n_index].item())
            gb_dataset['logits'].append(b_sorted_class_i_logits[n].item())
            gb_dataset['gb_label'].append(1.0)

        # 寻找最坏的100个同类图像
        w_sorted_class_i_logits, w_index = torch.sort(
            class_i_logits, descending=False)

        n = 0
        count_image = 0
        while count_image < 100:
            n_index = w_index[n]
            n_label = label_tensor[n_index].item()

            if int(n_label) == class_i:
                gb_dataset['data'].append(data_tensor[n_index, :, :, :].clone())
                gb_dataset['feature'].append(feature_tensor[n_index, :].clone())
                gb_dataset['imgname'].append(imgname_list[n_index])
                gb_dataset['origin_label'].append(label_tensor[n_index].item())
                gb_dataset['logits'].append(w_sorted_class_i_logits[n].item())
                gb_dataset['gb_label'].append(0.0)

                count_image += 1

            n += 1

    gb_dataset_file = open('save/gb_dataset/gb-dataset-small.pickle', 'wb')
    pickle.dump(gb_dataset, gb_dataset_file)
    gb_dataset_file.close()

    # # 得到供观察数据的json文件
    # # 计算并存储每一类最(不)相似的前100个图像
    # for class_i in range(64):
    #     class_i = int(class_i)
    #     class_i_best = {}
    #     class_i_worst = {}
    #     class_i_logits = logits_tensor[:, class_i]

    #     # 寻找最好的100个图像
    #     b_sorted_class_i_logits, b_index = torch.sort(class_i_logits, descending=True)

    #     for n in range(100):
    #         n_index = b_index[n]
    #         n_imgname = imgname_list[n_index]
    #         n_label = label_tensor[n_index].item()
    #         n_logits = b_sorted_class_i_logits[n].item()

    #         class_i_best[int(n)] = {
    #             'imgname': n_imgname,
    #             'label': int(n_label),
    #             'logits': n_logits
    #         }

    #     json_str = json.dumps(class_i_best)
    #     with open('save/dataset_info/best_images/train_class_' + str(class_i) + '_best.json', 'w') as json_file:
    #         json_file.write(json_str)

    #     # 寻找最坏的100个同类图像
    #     w_sorted_class_i_logits, w_index = torch.sort(class_i_logits, descending=False)

    #     n = 0
    #     count_image = 0
    #     while count_image < 100:
    #         n_index = w_index[n]
    #         n_imgname = imgname_list[n_index]
    #         n_label = label_tensor[n_index].item()
    #         n_logits = w_sorted_class_i_logits[n].item()

    #         if int(n_label) == class_i:
    #             class_i_worst[int(count_image)] = {
    #                 'imgname': n_imgname,
    #                 'label': int(n_label),
    #                 'logits': n_logits
    #             }

    #             count_image += 1

    #         n += 1

    #     json_str = json.dumps(class_i_worst)
    #     with open('save/dataset_info/worst_images/train_class_' + str(class_i) + '_worst.json', 'w') as json_file:
    #         json_file.write(json_str)

    if False:
        # 记录类别编号和名称匹配信息
        json_str = json.dumps(classname_dict)
        with open('save/dataset_info/train_class_info.json', 'w') as json_file:
            json_file.write(json_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./configs/get_best_base_data.yaml')
    parser.add_argument('--name', default='best_base_data')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    get_best_base_sample(config)
