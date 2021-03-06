import argparse
import torch
from models import build_model
from data import build_dataloader
import torch.nn.functional as F
import yaml
import utils
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

def main(config):
    # ===== 准备记录以及log信息 =====
    save_name = args.name
    save_path = os.path.join('./save/train_gb_classify_network', save_name)

    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    tb_writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    # ===== 准备数据、模型 =====
    # train data
    train_dataloader = build_dataloader(config['train_dataloader_args'])

    # val data
    val_dataloader = build_dataloader(config['val_dataloader_args'])

    # model
    gb_net_model = build_model(config['network_args'])
    utils.log('num params: {}'.format(utils.compute_n_params(gb_net_model)))

    # ===== 训练 =====

    # optimizer
    trainer_args = config['trainer_args']
    optimizer, lr_scheduler = utils.make_optimizer(gb_net_model.parameters(), trainer_args['optimizer_name'], **trainer_args['optimizer_args'])
    
    max_epoch = trainer_args['max_epoch']
    save_epoch = trainer_args['save_epoch']

    max_val_acc = 0.0

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    # 执行训练
    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        aves = {k: utils.Averager() for k in aves_keys}

        gb_net_model.train()
        tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, feature, gb_label in tqdm(train_dataloader, desc='train', leave=False):
            feature = feature.cuda()
            gb_label = gb_label.cuda()

            _, gb_logits = gb_net_model(feature)
            ce_loss = F.cross_entropy(gb_logits, gb_label)
            gb_acc = utils.compute_acc(gb_logits, gb_label)

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            aves['train_loss'].add(ce_loss.item())
            aves['train_acc'].add(gb_acc)

            gb_logits = None
            ce_loss = None
        
        # 验证
        gb_net_model.eval()
        for data, feature, gb_label in tqdm(val_dataloader, desc='val', leave=False):
            feature = feature.cuda()
            gb_label = gb_label.cuda()

            with torch.no_grad():
                _, gb_logits = gb_net_model(feature)
                ce_loss = F.cross_entropy(gb_logits, gb_label)
                gb_acc = utils.compute_acc(gb_logits, gb_label)

            aves['val_loss'].add(ce_loss.item())
            aves['val_acc'].add(gb_acc)


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

        log_str += ', val {:.4f}|{:.4f}'.format(aves['val_loss'], aves['val_acc'])
        tb_writer.add_scalars('loss', {'val': aves['val_loss']}, epoch)
        tb_writer.add_scalars('acc', {'val': aves['val_acc']}, epoch)


        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)

        utils.log(log_str)

        if config.get('_parallel'):
            model_ = gb_net_model.module
        else:
            model_ = gb_net_model

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
    parser.add_argument('--config', default='./configs/train_gb_classify_network.yaml')
    parser.add_argument('--name', default='train_gb_classify_network')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)

    main(config)
