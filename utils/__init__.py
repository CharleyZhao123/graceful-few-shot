import os
import shutil
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils.check_arguments import check_args
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)
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

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

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


def compute_acc(logits, label, reduction='mean'):
    # logits: [300, 5]
    _, sim_rank = logits.sort(descending=True, dim=1)
    # rank-1
    # ret = (torch.argmax(logits, dim=1) == label).float()  # [300]
    ret = (sim_rank[:, 0] == label).float()
    # rank-2
    # ret_1 = (sim_rank[:, 0] == label)
    # ret_2 = (sim_rank[:, 1] == label)
    # ret = (ret_1 | ret_2).float()
    # rank-4
    # ret_1 = (sim_rank[:, 0] == label)
    # ret_2 = (sim_rank[:, 1] == label)
    # ret_3 = (sim_rank[:, 2] == label)
    # ret_4 = (sim_rank[:, 3] == label)
    # ret = (ret_1 | ret_2 | ret_3 | ret_4).float()

    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
