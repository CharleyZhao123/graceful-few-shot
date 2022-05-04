import os
import argparse
import utils
utils.set_gpu('1')
import torch
from models import build_model

from data import build_dataloader
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter


def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/train_for_mushi', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    tb_writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 设定随机种子 =====
    utils.set_seed(771331)

    # ===== 准备数据、模型 =====
    # sim train data
    mix_train_dataloader = build_dataloader(
        config['mix_train_dataloader_args'])

    # sim train data
    sim_train_dataloader = build_dataloader(
        config['sim_train_dataloader_args'])

    # true train data
    true_train_dataloader = build_dataloader(
        config['true_train_dataloader_args'])

    # final train data
    train_dataloader = true_train_dataloader

    # sim val data
    # sim_val_dataloader = build_dataloader(config['sim_val_dataloader_args'])

    # true val data
    true_val_dataloader = build_dataloader(config['true_val_dataloader_args'])

    # model
    network_args = config['network_args']
    pretrain_model = build_model(network_args['model_name'], network_args['model_args'])
    utils.log('num params: {}'.format(utils.compute_n_params(pretrain_model)))

    # ===== 训练 =====

    # optimizer
    trainer_args = config['trainer_args']
    optimizer, lr_scheduler = utils.make_optimizer(pretrain_model.parameters(
    ), trainer_args['optimizer_name'], **trainer_args['optimizer_args'])

    max_epoch = trainer_args['max_epoch']
    save_epoch = trainer_args['save_epoch']

    # max_sim_val_acc = 0.0
    # max_sim_val_acc_epoch = 0

    max_true_val_acc = 0.0
    max_true_val_acc_epoch = 0

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    # 执行训练
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['train_loss', 'train_acc', 'sim_val_loss',
                     'sim_val_acc', 'true_val_loss', 'true_val_acc']
        aves = {k: utils.Averager() for k in aves_keys}

        pretrain_model.train()
        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for image, label in tqdm(train_dataloader, desc='train', leave=False):
            image = image.cuda()
            label = label.cuda()

            _, logits = pretrain_model(image)
            ce_loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            aves['train_loss'].add(ce_loss.item())
            aves['train_acc'].add(acc)

            logits = None
            ce_loss = None

        # 验证
        # sim
        # pretrain_model.eval()
        # for image, label in tqdm(sim_val_dataloader, desc='sim val', leave=False):
        #     image = image.cuda()
        #     label = label.cuda()

        #     with torch.no_grad():
        #         _, logits = pretrain_model(image)
        #         ce_loss = F.cross_entropy(logits, label)
        #         acc = utils.compute_acc(logits, label)

        #     aves['sim_val_loss'].add(ce_loss.item())
        #     aves['sim_val_acc'].add(acc)

        # true
        pretrain_model.eval()
        for image, label in tqdm(true_val_dataloader, desc='true val', leave=False):
            image = image.cuda()
            label = label.cuda()

            with torch.no_grad():
                _, logits = pretrain_model(image)
                ce_loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

            aves['true_val_loss'].add(ce_loss.item())
            aves['true_val_acc'].add(acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # 是否多余?
        for k, v in aves.items():
            aves[k] = v.item()

        # 记录log, 保存checkpoint
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        epoch_str = str(epoch)

        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
            epoch_str, aves['train_loss'], aves['train_acc'])
        tb_writer.add_scalars('loss', {'train': aves['train_loss']}, epoch)
        tb_writer.add_scalars('acc', {'train': aves['train_acc']}, epoch)

        # log_str += ', sim val {:.4f}|{:.4f}'.format(
        #     aves['sim_val_loss'], aves['sim_val_acc'])
        # tb_writer.add_scalars(
        #     'sim loss', {'sim val': aves['sim_val_loss']}, epoch)
        # tb_writer.add_scalars(
        #     'sim acc', {'sim val': aves['sim_val_acc']}, epoch)

        log_str += ', true val {:.4f}|{:.4f}'.format(
            aves['true_val_loss'], aves['true_val_acc'])
        tb_writer.add_scalars(
            'true loss', {'true val': aves['true_val_loss']}, epoch)
        tb_writer.add_scalars(
            'true acc', {'true val': aves['true_val_acc']}, epoch)

        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)

        utils.log(log_str)

        if config.get('_parallel'):
            model_ = pretrain_model.module
        else:
            model_ = pretrain_model

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

        # if aves['sim_val_acc'] > max_sim_val_acc:
        #     max_sim_val_acc = aves['sim_val_acc']
        #     max_sim_val_acc_epoch = epoch
        #     torch.save(save_obj, os.path.join(
        #         save_path, 'max-sim-val-acc.pth'))

        if aves['true_val_acc'] > max_true_val_acc:
            max_true_val_acc = aves['true_val_acc']
            max_true_val_acc_epoch = epoch
            torch.save(save_obj, os.path.join(
                save_path, 'max-true-val-acc.pth'))

        tb_writer.flush()

    # log_str = 'max sim val epoch: {}, acc: {}'.format(
    #     max_sim_val_acc_epoch, max_sim_val_acc)
    log_str += '\n'
    log_str += 'max true val epoch: {}, acc: {}'.format(
        max_true_val_acc_epoch, max_true_val_acc)

    utils.log(log_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_for_mushi.yaml')
    parser.add_argument('--name', default='train_for_mushi')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
