from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import model_register, build_model
import utils
import utils.few_shot as fs

@model_register('semantic-prorotype-network')
class SemanticPrototypeNetwork(nn.Module):
    def __init__(self, encoder_name='resnet12', encoder_args={}, encoder_load_para={},
                 pr_name='', pr_args={}, pr_load_para={}, vlm_name='clip', sp_info_path=''):
        super().__init__()

        if vlm_name == 'clip':



