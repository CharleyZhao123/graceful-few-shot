import torch.nn as nn
from ..model_init import model_register, build_model

@model_register('base-pretrain-network')
class BasePretrainNetwork(nn.Module):
    
    def __init__(self, encoder_name, encoder_args,
                 classifier_name, classifier_args):
        super().__init__()
        self.encoder = build_model(encoder_name, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = build_model(classifier_name, **classifier_args)

    def forward(self, x):
        feature = self.encoder(x)
        logits = self.classifier(feature)
        return feature, logits