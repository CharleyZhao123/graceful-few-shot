import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import argparse
import torch

import torch.nn.functional as F
import numpy as np
import yaml
import utils
import utils.few_shot as fs
from tqdm import tqdm
import scipy.stats
from models import build_model
from data import build_dataloader


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/test_mva', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 准备数据、模型 =====
    # dataloader
    test_dataloader = build_dataloader(config['test_dataloader_args'])

    # model
    network_args = config['network_args']
    model = build_model(network_args['model_name'], network_args['model_args'])

    # 准备备用mva model, 保证每个task都是初始的mva, 不受之前的影响
    mva_name = network_args['model_args']['mva_name']
    mva_args = network_args['model_args']['mva_args']
    if mva_args.get('update'):
        update_mva = mva_args['update']
    else:
        update_mva = False
    origin_mva_model = build_model(mva_name, mva_args)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # task信息
    test_dataloader_args = config['test_dataloader_args']
    task_per_batch = test_dataloader_args['batch_size']

    test_sampler_args = test_dataloader_args['sampler_args']
    way_num = test_sampler_args['way_num']
    shot_num = test_sampler_args['shot_num']
    query_num = test_sampler_args['query_num']

    # ===== 设定随机种子 =====
    utils.set_seed(1)

    # ===== 测试 =====
    test_epochs = args.test_epochs
    aves_keys = ['test_loss', 'test_acc']
    aves = {k: utils.Averager() for k in aves_keys}
    test_acc_list = []

    model.eval()

    for epoch in range(1, test_epochs + 1):
        for image, _, _ in tqdm(test_dataloader, leave=False):
            image = image.cuda()  # [320, 3, 224, 224]

            # 重载mva参数
            if update_mva:
                model.mva.load_state_dict(origin_mva_model.state_dict())
            
            with torch.no_grad():
                # [320, 5]: 320 = 4 x (5 x (1 + 15))
                logits = model(image)

                label = fs.make_nk_label(
                    way_num, query_num, task_per_batch).cuda()

                loss = F.cross_entropy(logits, label)

                acc = utils.compute_acc(logits, label)

                aves['test_loss'].add(loss.item(), len(image))
                aves['test_acc'].add(acc, len(image))

                test_acc_list.append(acc)

        log_str = 'test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
            epoch,
            aves['test_acc'].item() * 100,
            mean_confidence_interval(test_acc_list) * 100,
            aves['test_loss'].item()
        )
        utils.log(log_str)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_mva_network.yaml')
    parser.add_argument('--name', default='test_mva_network')
    parser.add_argument('--test-epochs', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
