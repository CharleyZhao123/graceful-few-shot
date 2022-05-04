import json
import os
import random

def get_gan_mushi_info(gan_base_path, subsets_info, dataset_info_json_path):
    info_dict = {
        "fathers": []
    }
    id = 0

    gan_base_list = os.listdir(gan_base_path)
    print(gan_base_list)

    for father, subsets in subsets_info.items():
        info = {
            "id": id,
            "name": father,
            "images": [],
            "num": 0,
        }
        id = id + 1

        images_list = []

        for s in subsets:
            if s not in gan_base_list:
                continue
            path = os.path.join(gan_base_path, s)
            file_list = os.listdir(path)
            for f in file_list:
                if '_paste' in f:
                    file_path = os.path.join(path, f)
                    images_list.append(file_path)
        
        random.shuffle(images_list)
        info["images"] = images_list
        info["num"] = len(images_list)

        info_dict["fathers"].append(info)

    json_str = json.dumps(info_dict)
    with open(dataset_info_json_path, 'w') as json_f:
        json_f.write(json_str)


if __name__ == '__main__':
    gan_base_path = '/space1/leoxing/mocod_results'
    subsets_info = {
    'person': ['girl', 'woman', 'man', 'walkingman', 'men', 'Ped1_2', 'Ped3_2', 'Man_2', 'men_2'],
    'carrier': [
        'VH_GAZ_Tiger_2',
        'VH_GAZ_Tiger_4',
        'TIGER',
        'MAZ537',
        'BTR152_2',
        'BTR152_3',
        'DJS-2022_2',
        'DJS-2022_5',
        'DJS-2022_8',
        'DJS-2022_11'
        'DJS2022_2'
    ],
    'tank': ['T-90', 'T-90_2', 'T-90_4', 'T-90A_2'],
    'armored': ['LAV_2', 'VH_BTR70_2', 'VH_BTR152_11', 'DV_LAV-Jackal_2', 'DV-LAV-Jackal_8'],
    'car': ['HatchBack_2', 'Sedan_2', 'Sedan2_2', 'sedan_4door_2', 'sedan2Door_2', 'sedan2Door', 'Hybrid_2']
    }

    dataset_info_json_path = 'gan_mushi_info.json'

    get_gan_mushi_info(gan_base_path, subsets_info, dataset_info_json_path)