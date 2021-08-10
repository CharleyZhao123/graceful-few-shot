import math

import torch
import torch.nn as nn

import models
import utils
import sys
sys.path.append('..')
from models import register


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super().__init__()
        self.linear = nn.Linear(in_dim, class_num)

    def forward(self, x):
        return self.linear(x)


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, class_num, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(class_num, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)

