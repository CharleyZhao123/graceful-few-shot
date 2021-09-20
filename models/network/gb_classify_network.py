import torch.nn as nn

import models
import sys
sys.path.append('..')
from models import register

@register('gb-classify-network')
class GBClassifyNetwork(nn.Module):
    def __init__(self, encoder_name, encoder_args, classifier_name, classifier_args):
        super().__init__()
        
        # skip_encoder: 跳过encoder, 直接使用提取好的特征
        self.skip_encoder = encoder_args['skip']
        if not self.skip_encoder:
            self.encoder = models.make(encoder_name, **encoder_args)
            classifier_args['in_dim'] = self.encoder.out_dim
        else:
            classifier_args['in_dim'] = 512

        self.classifier = models.make(classifier_name, **classifier_args)
    
    def forward(self, x):
        if not self.skip_encoder:
            feature = self.encoder(x)
        else:
            feature = x
        
        logits = self.classifier(feature)
        return feature, logits
