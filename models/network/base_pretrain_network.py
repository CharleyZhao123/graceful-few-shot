import torch.nn as nn

import models
import sys
sys.path.append('..')
from models import register

@register('base-pretrain-network')
class BasePretrainNetwork(nn.Module):
    
    def __init__(self, encoder_name, encoder_args,
                 classifier_name, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder_name, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier_name, **classifier_args)

    def forward(self, x):
        feature = self.encoder(x)
        logits = self.classifier(feature)
        return feature, logits