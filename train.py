import os
import argparse
import yaml

import utils

def do_train(config):
    # 
    pass

def main(config):
    sv_folder_path = os.path.join('./save', args.svname)
    utils.ensure_path(sv_folder_path)

    
    pass

if __name__ == '__main__':
    # build argument parse with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_pre_mini.yaml')
    parser.add_argument('--svname', default='train_dt', help='saved folder name')
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    # load config from yaml
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # set gpu(s)
    if len(args.gpu.split(',')) > 1:
        config['model_parallel'] = True
        config['all_gpu'] = args.gpu
    
    utils.set_gpu(args.gpu)

    main(config)