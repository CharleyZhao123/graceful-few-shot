from torch.utils.data import DataLoader
from . import datasets
from . import samplers
import sys
sys.path.append('..')
from utils import check_args


default_dataloader_args = {
    # 数据集名称 ('img-mini-imagenet', 'mini-imagenet')
    'dataset_name': 'img-mini-imagenet',
    'dataset_args': {
        'split': 'train',  # 数据集划分名称 ('train', 'val', 'test')
        'augment': 'default'
    },
    # 采样器名称 ('default-sampler', 'metatasks-sampler', 'sequential-sampler')
    'sampler_name': 'default-sampler',
    'batch_size': 128,  # 批大小 (36, 48, 64, 128) (1, 4)
}
default_sampler_args = {
    'batch_num': 200,
    'shot_num': 1,
    'way_num': 5,
    'query_num': 15
}


def build_dataloader(dataloader_args=default_dataloader_args):
    '''
    构建dataloader
    输入: dataloader_args
    输出: 符合参数设定的dataloader
    '''

    # ===== 检查默认必备参数是否具备, 否则对其进行设定 =====
    dataloader_args = check_args(default_dataloader_args, dataloader_args)

    # ===== 设定dataset =====
    dataset_name = dataloader_args['dataset_name']
    dataset_args = dataloader_args['dataset_args']
    dataset = datasets.make(dataset_name, dataset_args)

    # ===== 设定sampler, dataloader =====
    if dataloader_args['sampler_name'] == 'metatasks-sampler':
        if not dataloader_args.get['sampler_args']:
            sampler_args = default_dataloader_args
        else:
            sampler_args = check_args(
                default_sampler_args, dataloader_args['sampler_args'])
        sampler = samplers.MetatasksSampler(dataset.label, sampler_args['batch_num'],
                                            sampler_args['way_num'], sampler_args['shot_num'] +
                                            sampler_args['query_num'],
                                            ep_per_batch=dataloader_args['batch_size'])

        dataloader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)
    elif dataloader_args['sampler_name'] == 'sequential-sampler':
        dataloader = DataLoader(
            dataset, dataloader_args['batch_size'], shuffle=False, num_workers=8, pin_memory=True)        
    else:
        dataloader = DataLoader(
            dataset, dataloader_args['batch_size'], shuffle=True, num_workers=8, pin_memory=False)
    
    return dataloader


if __name__ == '__main__':

    dataloader_args = {
        'dataset_args': {
            'name': 'img-mini-imagenet',
            'split': 'test',
        },
        'batch_size': 64,
    }

    build_dataloader(dataloader_args)
