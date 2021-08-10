import models
import torch
import sys
sys.path.append('..')
from utils import check_args


def build_model(network_args=None):
    '''
    输入: 神经网络模型配置参数
    输出: 初始化好参数(根据存储的参数或者随机初始化参数)的神经网络模型
    '''

    model_name = network_args['model_name']
    model_args = network_args['model_args']
    model_load_para = network_args['model_load_para']

    if model_load_para.get('load'):
        model_para = torch.load(model_load_para['load'])
        model = models.load(model_para)
    else:
        model = models.make(model_name, **model_args)

    if model_load_para.get('load_encoder'):
        encoder_para = torch.load(model_load_para['load_encoder'])
        encoder = models.load(encoder_para).encoder
        model.encoder.load_state_dict(encoder.state_dict())

    if model_load_para.get('load_classifier'):
        classifier_para = torch.load(model_load_para['load_classifier'])
        classifier = models.load(classifier_para).classifier
        model.classifier.load_state_dict(classifier.state_dict())

    return model


if __name__ == '__main__':

    build_model()
