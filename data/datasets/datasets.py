import os

DEFAULT_ROOT = '/space1/zhaoqing/dataset/fsl'

datasets = {}


def model_register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_name, dataset_args):
    if dataset_args.get('root_path') is None:
        dataset_args['root_path'] = os.path.join(DEFAULT_ROOT, dataset_name)
    dataset = datasets[dataset_name](**dataset_args)
    return dataset
