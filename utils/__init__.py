import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.check_arguments import check_args
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

# set global gpu


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# make sure the path is valid, and create it


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))  # sv_folder_path
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                       or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


# set global log path
_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path

# make optimizer


def make_optimizer(params, name, lr, weight_decay=None,
                   milestones=None, gamma=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones, gamma)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler

# compute logits


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp
