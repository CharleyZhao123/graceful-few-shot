import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
from tensorboardX import SummaryWriter


def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/meta_train_mva', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    tb_writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 准备数据、模型 =====
    # dataloader
    train_dataloader = build_dataloader(config['train_dataloader_args'])
    val_dataloader = build_dataloader(config['val_dataloader_args'])

    # model
    network_args = config['network_args']
    model = build_model(network_args['model_name'], network_args['model_args'])

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # task信息
    train_dataloader_args = config['train_dataloader_args']
    task_per_batch = train_dataloader_args['batch_size']  # 默认1

    train_sampler_args = train_dataloader_args['sampler_args']
    way_num = train_sampler_args['way_num']
    shot_num = train_sampler_args['shot_num']
    query_num = train_sampler_args['query_num']

    # ===== 设定随机种子 =====
    utils.set_seed(1)

    # ===== 训练 =====
    # optimizer
    trainer_args = config['trainer_args']
    outer_optimizer, outer_lr_scheduler = utils.make_optimizer(model.parameters(), trainer_args['outer_optimizer_name'], **trainer_args['outer_optimizer_args'])

    full_epoch = trainer_args['full_epoch']
    save_epoch = trainer_args['save_epoch']

    max_val_acc = 0.0

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    # 执行训练
    for epoch in range(1, full_epoch + 1):
        timer_epoch.s()
        aves_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        aves = {k: utils.Averager() for k in aves_keys}

        model.train()
        tb_writer.add_scalar('lr', outer_optimizer.param_groups[0]['lr'], epoch)

        for image, _, _ in tqdm(train_dataloader, desc='train', leave=False):
            image = image.cuda()  # No Patch: [320, 3, 80, 80]; Patch: [320, 10, 3, 80, 80]
            label = fs.make_nk_label(
                way_num, query_num, task_per_batch).cuda()

            # [320, 5]: 320 = 4 x (5 x (1 + 15))
            logits = model(image)

            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            outer_optimizer.zero_grad()
            loss.backward()
            outer_optimizer.step()

            aves['train_loss'].add(loss.item())
            aves['train_acc'].add(acc)

            logits = None
            loss = None
        
        # 验证
        # 准备备用mva model, 保证每个task都是初始的mva, 不受之前的影响
        mva_args = network_args['model_args']['mva_args']
        if mva_args.get('update'):
            update_mva = mva_args['update']
        else:
            update_mva = False

        origin_mva_state_dict = model.mva.state_dict()

        model.eval()
        for image, _, _ in tqdm(val_dataloader, leave=False):
            image = image.cuda()  # No Patch: [320, 3, 80, 80]; Patch: [320, 10, 3, 80, 80]

            # 重载mva参数
            if update_mva:
                model.mva.load_state_dict(origin_mva_state_dict)
            
            with torch.no_grad():
                # [320, 5]: 320 = 4 x (5 x (1 + 15))
                logits = model(image)

                label = fs.make_nk_label(
                    way_num, query_num, task_per_batch).cuda()

                loss = F.cross_entropy(logits, label)

                acc = utils.compute_acc(logits, label)

                aves['val_loss'].add(loss.item())
                aves['val_acc'].add(acc)      

        # 重载mva参数
        if update_mva:
            model.mva.load_state_dict(origin_mva_state_dict)
            
        if outer_lr_scheduler is not None:
            outer_lr_scheduler.step()

        # 记录log, 保存checkpoint
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * full_epoch)

        str_epoch = str(epoch)
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                str_epoch, aves['train_loss'], aves['train_acc'])
        tb_writer.add_scalars('loss', {'train': aves['train_loss']}, epoch)
        tb_writer.add_scalars('acc', {'train': aves['train_acc']}, epoch)

        log_str += ', val {:.4f}|{:.4f}'.format(aves['val_loss'], aves['val_acc'])
        tb_writer.add_scalars('loss', {'val': aves['val_loss']}, epoch)
        tb_writer.add_scalars('acc', {'val': aves['val_acc']}, epoch)


        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)

        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = config['trainer_args']
        save_obj = {
            'file': __file__,
            'config': config,
            'model_sd': model_.state_dict(),
            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(
                save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['val_acc'] > max_val_acc:
            max_val_acc = aves['val_acc']
            torch.save(save_obj, os.path.join(save_path, 'max-val-acc.pth'))

        tb_writer.flush()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_meta_mva_network_task.yaml')
    parser.add_argument('--name', default='train_mva_network')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
