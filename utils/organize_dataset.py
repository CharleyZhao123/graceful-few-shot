import json
import csv
from json import encoder

def organize_dataset(input_info, input_dataset_info_path):
    '''
    为save/dataset_info/train_class_info.json增加中文名称字段
    todo: 目前保存的是Unicode码, 可修改为对应汉字
    '''
    output_info = {}
    for n, label in input_info.items():
        class_info = {
            "label": label
        }
        dataset_info_reader = csv.reader(open(input_dataset_info_path))
        for line in dataset_info_reader:
            if str(line[1]) == str(n):
                class_info['chinese_name'] = line[2]
                break
        output_info[n] = class_info

    return output_info


if __name__ == '__main__':
    input_info_path = '/space1/zhaoqing/code/graceful-few-shot/save/dataset_info/train_class_info.json'
    input_dataset_info_path = '/space1/zhaoqing/code/graceful-few-shot/save/imagenet_chinesename.csv'
    output_info_path = '/space1/zhaoqing/code/graceful-few-shot/save/dataset_info/train_class_info_plus.json'

    with open(input_info_path, 'r') as load_f:
        input_info = json.load(load_f)

    output_info = organize_dataset(input_info, input_dataset_info_path)

    json_str = json.dumps(output_info)
    with open(output_info_path, 'w') as json_f:
        json_f.write(json_str)
