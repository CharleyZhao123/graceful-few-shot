

def check_args(default_args, input_args):
    '''
    检查默认必备参数是否具备, 否则对其进行设定
    输入:
        default_args: 默认参数词典
        input_args: 输入的参数词典
    输出:
        new_args: 检查并纠正后的参数词典
    '''

    # 单独处理value为dict的情况
    for k, v in default_args.items():
        if type(v).__name__ == 'dict':
            if input_args.get(k):
                input_args[k] = check_args(v, input_args[k])
            else:
                input_args[k] = v
    
    # 合并default_args input_args, key相同则input_args覆盖dafault_args
    new_args = default_args.copy()
    new_args.update(input_args)

    return(new_args)


if __name__ == '__main__':
    default_dataloader_args = {
        'dataset_args': {
            'name': 'img-mini-imagenet',  # 数据集名称 ('img-mini-imagenet', 'mini-imagenet')
            'split': 'train',  # 数据集划分名称 ('train', 'val', 'test')
            'augment': 'default'
        },
        'sampler': 'default-sampler',  # 采样器名称 ('default-sampler', 'meta-sampler')
        'batch_size': '128',  # 批大小 (36, 48, 64, 128) (1, 4)
    }

    dataloader_args = {
        'dataset_args': {
            'name': 'img-mini-imagenet',
            'split': 'test',
        },
        'batch_size': '64',
    }

    new_args = check_args(default_dataloader_args, dataloader_args)
    print(new_args)