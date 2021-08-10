import torch
import torch.nn as nn
import os
import yaml
import json
import utils
import argparse
from data import build_dataloader
from models import build_model
from tqdm import tqdm

base_proto_f_path = 'visualization/saved_data/base_prototype_feature.pth'
base_proto_l_path = 'visualization/saved_data/base_prototype_label.pth'


def get_best_base_sample(config):

    # ===== 准备数据、模型 =====

    # nn classifier base prototype
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
    logits_tensor = torch.zeros([1, 64]).cuda()

    for data, label, imgname in tqdm(train_dataloader, leave=False):
        data = data.cuda()
        if False:
            # 记录类别编号和名称匹配信息
            if imgname[0][0:9] not in classname_dict:
                classname_dict[imgname[0][0:9]] = label[0].item()
        with torch.no_grad():
            logits = model(data)

        imgname_list.append(imgname[0])
        label_tensor = torch.cat((label_tensor, label), 0)
        logits_tensor = torch.cat((logits_tensor, logits), 0)

    label_tensor = label_tensor[1:]  # [38400]
    logits_tensor = logits_tensor[1:, :]  # [38400, 64]

    # 计算并存储每一类最相似的前100个图像
    for class_i in range(64):
        class_i = int(class_i)
        class_i_best = {}
        class_i_worst = {}
        class_i_logits = logits_tensor[:, class_i]

        # 寻找最好的100个图像
        # sorted_class_i_logits, index = torch.sort(class_i_logits, descending=True)
        
        # for n in range(100):
        #     n_index = index[n]
        #     n_imgname = imgname_list[n_index]
        #     n_label = label_tensor[n_index].item()
        #     n_logits = sorted_class_i_logits[n].item()

        #     class_i_best[int(n)] = {
        #         'imgname': n_imgname,
        #         'label': n_label,
        #         'logits': n_logits
        #     }

        # json_str = json.dumps(class_i_best)
        # with open('save/dataset_info/best_images/train_class_' + str(class_i) + '_best.json', 'w') as json_file:
        #     json_file.write(json_str)
        
        # 寻找最坏的100个同类图像
        sorted_class_i_logits, index = torch.sort(class_i_logits, descending=False)
        
        n = 0
        count_image = 0
        while count_image < 100:
            n_index = index[n]
            n_imgname = imgname_list[n_index]
            n_label = label_tensor[n_index].item()
            n_logits = sorted_class_i_logits[n].item()

            if int(n_label) == class_i:
                class_i_worst[int(count_image)] = {
                    'imgname': n_imgname,
                    'label': n_label,
                    'logits': n_logits
                }

                count_image += 1
            
            n += 1

        json_str = json.dumps(class_i_worst)
        with open('save/dataset_info/worst_images/train_class_' + str(class_i) + '_worst.json', 'w') as json_file:
            json_file.write(json_str)


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
