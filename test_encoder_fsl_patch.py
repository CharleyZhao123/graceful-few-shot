import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    save_path = os.path.join('./save/test_encoder_fsl', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 准备数据、模型 =====
    # dataloader
    test_dataloader = build_dataloader(config['test_dataloader_args'])

    # model
    network_args = config['network_args']
    model = build_model(network_args['model_name'], network_args['model_args'], network_args['model_load_para'])
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
            image = image.cuda()  # [320, 10, 3, 80, 80]

            with torch.no_grad():
                # ===== task info =====
                image_num = image.shape[0]
                patch_num = image.shape[1]

                # ===== 处理图像以及特征 =====
                # [320x10, 3, 80, 80]
                image = image.reshape(-1, image.shape[2], image.shape[3], image.shape[4])

                # [320x10, 512]
                if 'encode_image' in dir(model):
                    image_feature = model.encode_image(image).float()
                else:
                    image_feature = model(image)
                
                feat_dim = image_feature.shape[-1]
                
                image_feature = image_feature.reshape(image_num, patch_num, feat_dim)  # [320, 10, 512]

                # 划分并变换shot和query
                # shot_feat: [4, 5, 1, P, 512]
                # query_feat: [4, 75, P, 512]
                shot_feat, query_feat = fs.split_shot_query(
                    image_feature, way_num, shot_num, query_num, task_per_batch)
                
                new_shot_num = shot_num * patch_num
                # 相当于将S shot变换为SxP shot: [T, W, S, P, dim] -> [T, W, SxP, dim]
                shot_feat = shot_feat.reshape(task_per_batch, way_num, new_shot_num, feat_dim)
                # [T, Q, P, dim] -> [T, Q, dim]
                query_feat = query_feat[:, :, 0, :]                

                # ===== 计算相似度和logits, 得到结果 =====
                if config['similarity_method'] == 'cos':
                    shot_feat = shot_feat.mean(dim=-2)
                    shot_feat = F.normalize(shot_feat, dim=-1)
                    query_feat = F.normalize(query_feat, dim=-1)
                    metric = 'dot'
                elif config['similarity_method'] == 'sqr':
                    shot_feat = shot_feat.mean(dim=-2)
                    metric = 'sqr'

                logits = utils.compute_logits(
                    query_feat, shot_feat, metric=metric, temp=1.0).view(-1, way_num)

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
    parser.add_argument('--config', default='./configs/test_encoder_fsl_patch.yaml')
    parser.add_argument('--name', default='test_encoder_fsl')
    parser.add_argument('--test-epochs', type=int, default=1)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
