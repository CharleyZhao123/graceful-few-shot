import torch.nn as nn

import models
import sys
sys.path.append('..')
from models import register

@register('mva-natwork')
class MVANetwork(nn.Module):
    def __init__(self, encoder_name, encoder_args, mva_name, mva_args):
        super().__init__()
        self.encoder = models.make(encoder_name, **encoder_args)
        self.mva = models.make(mva_name, mva_args)
    
    def forward(self, x):
        pass

