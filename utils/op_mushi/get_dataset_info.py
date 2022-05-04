import json
import os
from posixpath import join

def get_dataset_info(dataset_path, dataset_info_json_path):

    file_list = os.listdir(dataset_path)

    with open(dataset_info_json_path, 'r') as json_f:
        info_dict = json.load(json_f)
    print(len(file_list))
    # 统计图像数据信息
    for f in file_list:
        if 'json' in f:
            label_json_path = os.path.join(dataset_path, f)
            with open(label_json_path, 'r') as json_f:
                label_dict = json.load(json_f)
            label_list = label_dict['labels']['label_list']

            if len(label_list) != 1:
                print(f)

            l = label_list[0]
            l_type = l['type']
            l_flag = False

            for c in info_dict['categories']:
                if c['name'] in l_type:
                    l_flag = True
                    if c.get('images'):
                        c['images'].append(f)
                        c['num'] += 1
                        c['other_name'] = set(c['other_name'])
                        c['other_name'].add(l_type)
                        c['other_name'] = list(c['other_name'])
                    else:
                        c['images'] = [f]
                        c['num'] = 1
                        c['other_name'] = set()
                        c['other_name'].add(l_type)
                        c['other_name'].add(c['name'])
                        c['other_name'] = list(c['other_name'])
            
            if not l_flag:
                print(l_type)


    # 统计父类信息
    father_list = info_dict['fathers']
    for f in father_list:
        f_name = f['name']
        f['images'] = []
        f['num'] = 0
        for c in info_dict['categories']:

            if c.get('images') == None:
                continue

            if c['father'] == f_name:
                f['images'] += c['images']
                f['num'] += c['num']
        f['images'] = list(set(f['images']))
        f['num'] = len(f['images'])
    
    # 父类信息查重
    for f in father_list:
        f_images_list = f['images']
        for f_o in father_list:
            if f_o == f:
                continue
            f_o_images_list = f_o['images']
            # 求交集
            inter_list = list(set(f_images_list).intersection(set(f_o_images_list)))
            if len(inter_list) > 0:
                print(f['name'])
                print(f_o['name'])
                print(inter_list)
      

    json_str = json.dumps(info_dict)
    new_json_path = dataset_info_json_path.replace('dataset', 'new_dataset')
    with open(new_json_path, 'w') as json_f:
        json_f.write(json_str)

if __name__ == '__main__':
    dataset_path = '/Users/charleyzhao/code/dataset/animals_origin/mushi'
    dataset_info_json_path = 'dataset_info.json'

    get_dataset_info(dataset_path, dataset_info_json_path)
