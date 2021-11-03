# 施工ing

import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from data import build_dataloader
from sklearn.metrics import roc_auc_score

import models
from models.build_model import build_model
import utils
import utils.few_shot as fs


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/test_fsl', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))


    # ===== 准备数据、模型 =====
    # dataloader
    test_dataloader = build_dataloader(config['test_dataloader_args'])

    # model
    encoder_model = build_model(config['network_args'])
    utils.log('num params: {}'.format(utils.compute_n_params(encoder_model)))

    encoder_model.eval()

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            with torch.no_grad():
                if not args.sauc:
                    logits = model(x_shot, x_query).view(-1, n_way)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                            *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(
                            bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--test-epochs', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)