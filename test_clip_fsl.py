import argparse
import torch

from data import build_dataloader
import clip
import torch.nn.functional as F
import numpy as np
import yaml
import utils
import utils.few_shot as fs
import os
from tqdm import tqdm
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/test_clip_fsl', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 准备数据、模型 =====
    # dataloader
    test_dataloader = build_dataloader(config['test_dataloader_args'])

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # task信息
    test_dataloader_args = config['test_dataloader_args']
    task_per_batch = test_dataloader_args['batch_size']

    test_sampler_args = test_dataloader_args['sampler_args']
    way_num = test_sampler_args['way_num']
    shot_num = test_sampler_args['shot_num']
    query_num = test_sampler_args['query_num']

    # ===== 测试 =====
    test_epochs = args.test_epochs
    aves_keys = ['test_loss', 'test_acc']
    aves = {k: utils.Averager() for k in aves_keys}
    test_acc_list = []

    model.eval()
    np.random.seed(0)

    for epoch in range(1, test_epochs + 1):
        for image, _, _ in tqdm(test_dataloader, leave=False):
            image = image.cuda()  # [320, 3, 224, 224]

            with torch.no_grad():
                # [320, 512]: 320 = 4 x (5 x (1 + 15))
                image_feature = model.encode_image(image)

                # 划分shot和query
                # x_shot: [4, 5, 1, 512]
                # x_query: [4, 75, 512]
                x_shot, x_query = fs.split_shot_query(
                    image_feature, way_num, shot_num, query_num, task_per_batch)

                # 计算相似度和logits
                if config['similarity_method'] == 'cos':
                    x_shot = x_shot.mean(dim=-2)
                    x_shot = F.normalize(x_shot, dim=-1)
                    x_query = F.normalize(x_query, dim=-1)
                    metric = 'dot'
                elif config['similarity_method'] == 'sqr':
                    x_shot = x_shot.mean(dim=-2)
                    metric = 'sqr'

                logits = utils.compute_logits(
                    x_query, x_shot, metric=metric, temp=1.0).view(-1, way_num)

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
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_clip_fsl.yaml')
    parser.add_argument('--name', default='test_clip_fsl')
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
