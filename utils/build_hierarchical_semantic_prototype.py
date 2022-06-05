import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import clip
from PIL import Image
import os
import pandas as pd
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_prompts = {
    'adj_1': 'A [] photo',
    'n_1': 'A photo of []',
    'n_2': 'A photo has []'
}

low_level_prompts = [
    base_prompts['adj_1'].replace('[]', 'red'),
    base_prompts['adj_1'].replace('[]', 'green'),
    base_prompts['adj_1'].replace('[]', 'blue'),
    base_prompts['adj_1'].replace('[]', 'yellow'),
    base_prompts['adj_1'].replace('[]', 'white'),
    base_prompts['adj_1'].replace('[]', 'black'),
]

middle_level_prompts = [
    base_prompts['n_2'].replace('[]', 'a single object'),
    base_prompts['n_2'].replace('[]', 'many objects'),
    base_prompts['adj_1'].replace('[]', 'simple'),
    base_prompts['adj_1'].replace('[]', 'complex')

]

high_level_prompts = [
    base_prompts['n_1'].replace('[]', 'person'),
    base_prompts['n_1'].replace('[]', 'animal'),
    base_prompts['n_1'].replace('[]', 'water'),
    base_prompts['n_1'].replace('[]', 'metal'),
    base_prompts['n_1'].replace('[]', 'fire'),
    base_prompts['n_1'].replace('[]', 'wood'),
    base_prompts['n_1'].replace('[]', 'ground'),
]

def get_level_prototype(prompt_list, save_path):
    with torch.no_grad():
        text = clip.tokenize(prompt_list).to(device)
        text_features = model.encode_text(text)

        torch.save(text_features, save_path)


def build_hs_prototype(path):

    # low
    low_save_path = os.path.join(path, 'low_proto.pth')
    # get_level_prototype(low_level_prompts, low_save_path)

    # middle
    middle_save_path = os.path.join(path, 'middle_proto.pth')
    # get_level_prototype(middle_level_prompts, middle_save_path)

    # high
    high_save_path = os.path.join(path, 'high_proto.pth')
    # get_level_prototype(high_level_prompts, high_save_path)

    # info
    save_info_path = os.path.join(path, 'info.json')
    info_dict = {
        'low': {
            'prompt': low_level_prompts,
            'path': low_save_path,
        },
        'middle': {
            'prompt': middle_level_prompts,
            'path': middle_save_path,
        },
        'high': {
            'prompt': high_level_prompts,
            'path': high_save_path
        }
    }

    json_str = json.dumps(info_dict)
    with open(save_info_path, 'w') as json_file:
        json_file.write(json_str)


if __name__ == '__main__':
    path = '/space1/zhaoqing/code/graceful-few-shot/save/semantic_prototype'
    build_hs_prototype(path)
